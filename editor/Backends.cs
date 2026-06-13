// Backends.cs — the editor talks to "a robot brain" through one tiny interface.
// NativeBackend drives the real DX12/MuJoCo engine; DemoBackend is a built-in
// pretend brain so the app is fully playable before the native engine is built.
using System;
using System.IO;
using System.Threading;
using Aistank.Training;

namespace Aistank.Editor;

public enum RobotMode { Idle, Training, Playing }

public readonly record struct RobotStatus(
    RobotMode Mode,
    float Skill,            // 0 = falls instantly, 1 = walking champion
    double StepsPerSecond,  // 0 in demo mode
    long TotalSteps);

public interface IRobotBackend : IDisposable
{
    bool IsReal { get; }
    void StartTraining();
    void StartPlaying();
    void Stop();
    RobotStatus GetStatus();

    /// <summary>Real MuJoCo pose of one agent: body world positions + parent indices.
    /// Returns false when no physics is available (demo mode).</summary>
    bool TryGetPose(out float[] bodyXyz, out int[] parents);
}

// ---------------------------------------------------------------- DemoBackend

/// <summary>Pretend brain: skill grows while training, freezes while playing.</summary>
public sealed class DemoBackend : IRobotBackend
{
    private RobotMode _mode = RobotMode.Idle;
    private float _skill;
    private long _steps;
    private DateTime _last = DateTime.UtcNow;

    public bool IsReal => false;

    public void StartTraining() { _mode = RobotMode.Training; _last = DateTime.UtcNow; }
    public void StartPlaying()  { _mode = RobotMode.Playing; }
    public void Stop()          { _mode = RobotMode.Idle; }

    public RobotStatus GetStatus()
    {
        var now = DateTime.UtcNow;
        float dt = (float)(now - _last).TotalSeconds;
        _last = now;
        if (_mode == RobotMode.Training)
        {
            // Logistic-ish learning curve: fast early gains, slow mastery.
            _skill = Math.Min(1f, _skill + dt * 0.03f * (1.1f - _skill));
            _steps += (long)(dt * 250_000);   // pretend throughput
        }
        return new RobotStatus(_mode, _skill, _mode == RobotMode.Training ? 250_000 : 0, _steps);
    }

    public bool TryGetPose(out float[] bodyXyz, out int[] parents)
    {
        bodyXyz = []; parents = [];
        return false;
    }

    public void Dispose() { }
}

// -------------------------------------------------------------- NativeBackend

/// <summary>
/// Real brain: owns the native engine + PPO trainer on a background thread.
/// Skill shown to the user is the running-normalized mean reward.
/// </summary>
public sealed class NativeBackend : IRobotBackend
{
    private readonly HumanoidEnvironment _env;
    private readonly PpoTrainer _trainer;
    private Thread? _worker;
    private volatile RobotMode _mode = RobotMode.Idle;
    private volatile bool _running;

    private float _skill;
    private float _rewardMin = float.MaxValue, _rewardMax = float.MinValue;
    private double _stepsPerSec;
    private long _totalSteps;

    public bool IsReal => true;

    private readonly int[] _bodyParents;
    private readonly float[] _poseBuffer;

    public NativeBackend(string mjcfPath)
    {
        if (!File.Exists(mjcfPath))
            throw new FileNotFoundException(mjcfPath);
        _env = new HumanoidEnvironment(new EnvironmentOptions
        {
            MjcfPath = mjcfPath,
            // 1024 agents keeps the editor's tick + PPO update snappy; the
            // headless CLI sample runs the full 4096-agent batch.
            NumAgents = 1024,
            RolloutHorizon = 32,
        });
        _trainer = new PpoTrainer(_env.ObservationDim, _env.ActionDim, _env.PolicyParamCount);
        _env.SetPolicyWeights(_trainer.CurrentWeights);
        _bodyParents = _env.GetBodyParents();
        _poseBuffer = new float[_env.BodyCount * 3];
    }

    public bool TryGetPose(out float[] bodyXyz, out int[] parents)
    {
        parents = _bodyParents;
        bodyXyz = _poseBuffer;
        return _env.TryGetAgentBodyPositions(0, _poseBuffer);
    }

    public void StartTraining() => StartLoop(RobotMode.Training);
    public void StartPlaying()  => StartLoop(RobotMode.Playing);

    private void StartLoop(RobotMode mode)
    {
        Stop();
        _mode = mode;
        _running = true;
        _worker = new Thread(Loop) { IsBackground = true, Name = "aistank-loop" };
        _worker.Start();
    }

    public void Stop()
    {
        _running = false;
        _worker?.Join();
        _worker = null;
        _mode = RobotMode.Idle;
    }

    private void Loop()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        long stepsAtStart = _totalSteps;
        while (_running)
        {
            for (int t = 0; t < 32 && _running; t++)
            {
                _env.Tick();
                _totalSteps += _env.NumAgents;
            }
            if (!_running) break;

            var rollout = _env.MapRollout();
            float meanReward;
            if (_mode == RobotMode.Training)
            {
                var update = _trainer.Update(rollout);
                _env.SetPolicyWeights(_trainer.CurrentWeights);
                meanReward = update.MeanReward;
            }
            else
            {
                float sum = 0;
                foreach (float r in rollout.Rewards) sum += r;
                meanReward = sum / rollout.Rewards.Length;
            }

            // Map reward into a 0..1 "skill" the UI can show as stars.
            _rewardMin = Math.Min(_rewardMin, meanReward);
            _rewardMax = Math.Max(_rewardMax, meanReward);
            float span = Math.Max(_rewardMax - _rewardMin, 1e-3f);
            _skill = Math.Clamp((meanReward - _rewardMin) / span, 0f, 1f);

            _stepsPerSec = (_totalSteps - stepsAtStart) / sw.Elapsed.TotalSeconds;
        }
    }

    public RobotStatus GetStatus()
        => new(_mode, _skill, _stepsPerSec, _totalSteps);

    public void Dispose()
    {
        Stop();
        _env.Dispose();
    }
}
