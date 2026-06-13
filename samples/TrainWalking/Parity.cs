// Parity.cs — GPU-vs-CPU gradient parity self-test.
//
// Collects one real rollout, then computes the full-batch PPO gradient two ways:
//   • on the GPU  (Engine_RunGradientOnly → Engine_DownloadGrad)
//   • on the CPU  (PpoTrainer.ComputeFullBatchGradient, the trusted reference)
// and reports the largest element-wise disagreement. They should match to a few
// thousandths (fp32 reduction-order differences only).
using System;
using Aistank;
using Aistank.Training;

static class Parity
{
    public static void Run()
    {
        // Small batch so the CPU oracle is quick; horizon long enough for GAE.
        using var env = new HumanoidEnvironment(new EnvironmentOptions
        {
            MjcfPath = "assets/humanoid16.xml",
            NumAgents = 64,
            RolloutHorizon = 8,
        });
        env.InitGpuTrainer();

        var trainer = new PpoTrainer(env.ObservationDim, env.ActionDim, env.PolicyParamCount);
        env.InitPolicyWeights(trainer.CurrentWeights);   // GPU slot 0 == CPU weights

        // Fill every rollout cursor with real physics, then snapshot it.
        for (int t = 0; t < 8; t++) env.Tick();
        var rollout = env.MapRollout();

        int P = (int)env.PolicyParamCount;
        var gpuGrad = new float[P];
        env.GpuGradient(gpuGrad);                        // GPU forward/backward
        var cpuGrad = trainer.ComputeFullBatchGradient(rollout);

        double sumSqG = 0, sumSqC = 0, dot = 0, sumSqDiff = 0;
        for (int i = 0; i < P; i++)
        {
            double diff = gpuGrad[i] - (double)cpuGrad[i];
            sumSqDiff += diff * diff;
            sumSqG += gpuGrad[i] * (double)gpuGrad[i];
            sumSqC += cpuGrad[i] * (double)cpuGrad[i];
            dot += gpuGrad[i] * (double)cpuGrad[i];
        }
        double cosine = dot / (Math.Sqrt(sumSqG) * Math.Sqrt(sumSqC) + 1e-12);
        double relL2 = Math.Sqrt(sumSqDiff) / (Math.Sqrt(sumSqC) + 1e-12);

        Console.WriteLine($"params={P}  |grad_gpu|={Math.Sqrt(sumSqG):F3}  |grad_cpu|={Math.Sqrt(sumSqC):F3}");
        Console.WriteLine($"relative L2 diff = {relL2:E3}   cosine = {cosine:F8}");
        // fp32 reduction-order differences only — direction identical, magnitude within ~0.1%.
        Console.WriteLine(cosine > 0.99999 && relL2 < 2e-3
            ? "PARITY OK — GPU gradient matches the CPU reference."
            : "PARITY FAIL — investigate the GPU backward pass.");
    }
}
