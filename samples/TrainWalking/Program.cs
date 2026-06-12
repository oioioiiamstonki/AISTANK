// TrainWalking — headless training entry point (the editor app is the friendly UI;
// this is the keyboard-and-terminal version of the same loop).
// The engine keeps inference + reward + termination on the GPU; this layer only
// touches data once per rollout horizon (32 ticks) to run the PPO update.
using System;
using System.Diagnostics;
using Aistank;
using Aistank.Training;

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
