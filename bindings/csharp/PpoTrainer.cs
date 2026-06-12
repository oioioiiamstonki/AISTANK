// PpoTrainer.cs — trainer stub shared by the editor and the CLI sample.
// Owns the FP32 parameter vector whose layout matches CS_PolicyForward:
//   [W1 256×obs][b1 256][W2 256×256][b2 256][W3 (act+1)×256][b3 act+1]
// Update() is where GAE + clipped surrogate optimization goes — plug in
// TorchSharp, your own SIMD Adam, or (future) a CS_AdamStep compute kernel.
using System;

namespace Aistank.Training;

public sealed class PpoTrainer
{
    private readonly float[] _weights;
    public ReadOnlySpan<float> CurrentWeights => _weights;

    public PpoTrainer(uint obsDim, uint actDim, ulong paramCount)
    {
        _weights = new float[paramCount];
        // Orthogonal-ish init: per-layer scaled Gaussian noise, zero biases.
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

public readonly record struct UpdateResult(float MeanReward, float MeanEpisodeLength);
