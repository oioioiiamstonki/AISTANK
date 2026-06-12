// TrainWalking.cs — high-level training orchestration entry point.
// The engine keeps inference + reward + termination on the GPU; this layer only
// touches data once per rollout horizon (32 ticks) to run the PPO update.
using System;
using System.Diagnostics;
using Aistank;

const int ticksPerIteration = 32;   // must equal RolloutHorizon
const int iterations = 2000;

using var env = new HumanoidEnvironment(new EnvironmentOptions
{
    MjcfPath = "assets/humanoid16.xml",
    NumAgents = 4096,
    RolloutHorizon = ticksPerIteration,
    EnablePreview = false,
    RewardWeights = new RewardWeights { Forward = 1.25f, Alive = 0.5f, Energy = 0.005f, Upright = 0.3f },
});

Console.WriteLine($"AISTANK | agents={env.NumAgents} obs={env.ObservationDim} " +
                  $"act={env.ActionDim} params={env.PolicyParamCount}");

var trainer = new PpoTrainer(env.ObservationDim, env.ActionDim, env.PolicyParamCount);
env.SetPolicyWeights(trainer.CurrentWeights);

var sw = Stopwatch.StartNew();
for (int iter = 0; iter < iterations; iter++)
{
    float cpuMs = 0, gpuMs = 0;
    for (int t = 0; t < ticksPerIteration; t++)
    {
        var stats = env.Tick();
        cpuMs += stats.CpuPhysicsMs;
        gpuMs += stats.GpuComputeMs;
    }

    var rollout = env.MapRollout();             // single blocking readback per horizon
    var update = trainer.Update(rollout);       // GAE + PPO clip (stub — wire your optimizer)
    env.SetPolicyWeights(trainer.CurrentWeights);

    if (iter % 10 == 0)
    {
        double stepsPerSec = (double)env.NumAgents * ticksPerIteration * (iter + 1)
                             / sw.Elapsed.TotalSeconds;
        Console.WriteLine(
            $"iter {iter,5} | R̄ {update.MeanReward,8:F3} | epLen {update.MeanEpisodeLength,6:F0} " +
            $"| {stepsPerSec / 1e6,6:F2}M steps/s | cpu {cpuMs / ticksPerIteration,5:F2}ms " +
            $"| gpu {gpuMs / ticksPerIteration,5:F2}ms");
    }
}

/// <summary>
/// PPO trainer stub. Owns the FP32 parameter vector whose layout matches
/// CS_PolicyForward: [W1 256×obs][b1][W2 256×256][b2][W3 (act+1)×256][b3].
/// Update() is where GAE + clipped surrogate optimization goes — plug in
/// TorchSharp, your own SIMD Adam, or (future) a CS_AdamStep compute kernel.
/// </summary>
sealed class PpoTrainer
{
    private readonly float[] _weights;
    public ReadOnlySpan<float> CurrentWeights => _weights;

    public PpoTrainer(uint obsDim, uint actDim, ulong paramCount)
    {
        _weights = new float[paramCount];
        // Orthogonal-ish init: small uniform noise scaled per layer.
        var rng = new Random(42);
        int hidden = 256;
        int w1 = hidden * (int)obsDim;
        float s1 = MathF.Sqrt(2f / obsDim), s2 = MathF.Sqrt(2f / hidden), s3 = 0.01f;
        int i = 0;
        for (; i < w1; i++) _weights[i] = Gauss(rng) * s1;
        i += hidden;                                       // b1 = 0
        for (int e = i + hidden * hidden; i < e; i++) _weights[i] = Gauss(rng) * s2;
        i += hidden;                                       // b2 = 0
        for (; i < _weights.Length - ((int)actDim + 1); i++) _weights[i] = Gauss(rng) * s3;
    }

    public UpdateResult Update(RolloutView rollout)
    {
        // === PPO stub ===
        // 1. GAE over rollout.Rewards / rollout.Values / rollout.Dones
        // 2. Minibatch SGD on clipped surrogate + value loss + entropy bonus
        // 3. Write updated params into _weights
        float sum = 0;
        foreach (float r in rollout.Rewards) sum += r;
        return new UpdateResult(
            MeanReward: sum / rollout.Rewards.Length,
            MeanEpisodeLength: 0f);
    }

    private static float Gauss(Random rng)
        => MathF.Sqrt(-2f * MathF.Log(1f - (float)rng.NextDouble()))
           * MathF.Cos(2f * MathF.PI * (float)rng.NextDouble());
}

readonly record struct UpdateResult(float MeanReward, float MeanEpisodeLength);
