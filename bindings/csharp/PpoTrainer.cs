// PpoTrainer.cs — real PPO (clipped surrogate + GAE + Adam) over the GPU rollouts.
//
// Network (must match CS_PolicyForward exactly):
//   obs[O] → Dense(256) ELU → Dense(256) ELU → heads { mu[A] = tanh(z), V = z }
// Weight layout: [W1 256×O][b1 256][W2 256×256][b2 256][W3 (A+1)×256][b3 A+1],
// every W row-major per output neuron.
//
// The GPU samples actions as a ~ clamp(tanh(z) + σ·ε, -1, 1) with fixed σ;
// the trainer treats the policy as Normal(tanh(z), σ) and recomputes log-probs
// from the stored (obs, action) pairs, so CPU and GPU never disagree about
// which network produced the data: rollouts are always collected with the
// weights that were pushed after the previous Update().
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Aistank.Training;

public sealed class PpoTrainer
{
    // Hyperparameters (Isaac-Lab-ish defaults for humanoid locomotion).
    private const float Gamma = 0.99f;
    private const float Lambda = 0.95f;
    private const float ClipEps = 0.2f;
    private const float ValueCoef = 0.5f;
    private const float LearningRate = 3e-4f;
    private const float Sigma = 0.2f;            // must match the 0.2 in CS_PolicyForward
    private const int Epochs = 3;
    private const int MinibatchSize = 8192;
    private const int Hidden = 256;

    private readonly int _obsDim, _actDim;
    private readonly float[] _w;                 // parameters
    private readonly float[] _adamM, _adamV;
    private int _adamT;

    // Weight-layout offsets.
    private readonly int _w1, _b1, _w2, _b2, _w3, _b3;

    public ReadOnlySpan<float> CurrentWeights => _w;

    public PpoTrainer(uint obsDim, uint actDim, ulong paramCount)
    {
        _obsDim = (int)obsDim;
        _actDim = (int)actDim;
        _w = new float[paramCount];
        _adamM = new float[paramCount];
        _adamV = new float[paramCount];

        _w1 = 0;
        _b1 = _w1 + Hidden * _obsDim;
        _w2 = _b1 + Hidden;
        _b2 = _w2 + Hidden * Hidden;
        _w3 = _b2 + Hidden;
        _b3 = _w3 + (_actDim + 1) * Hidden;
        if (_b3 + _actDim + 1 != (int)paramCount)
            throw new ArgumentException($"param layout mismatch: {_b3 + _actDim + 1} vs {paramCount}");

        // He-style init for ELU layers, small final head.
        var rng = new Random(42);
        FillGauss(rng, _w1, Hidden * _obsDim, MathF.Sqrt(2f / _obsDim));
        FillGauss(rng, _w2, Hidden * Hidden, MathF.Sqrt(2f / Hidden));
        FillGauss(rng, _w3, (_actDim + 1) * Hidden, 0.01f);
    }

    private void FillGauss(Random rng, int offset, int count, float scale)
    {
        for (int i = 0; i < count; i++)
        {
            float u1 = 1f - (float)rng.NextDouble(), u2 = (float)rng.NextDouble();
            _w[offset + i] = scale * MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
        }
    }

    public UpdateResult Update(RolloutView rollout)
    {
        int T = rollout.Horizon, N = rollout.NumAgents, O = rollout.ObsDim, A = rollout.ActDim;
        if (O != _obsDim || A != _actDim) throw new ArgumentException("rollout dims mismatch");

        // ---- 1. Copy the SoA rollout into per-sample-major arrays --------------
        // GPU layout: obs[(t*O + o)*N + n]. CPU wants obs[sample][o], sample = t*N+n.
        int S = T * N;
        var obs = new float[S * O];
        var act = new float[S * A];
        var rew = new float[S];
        var val = new float[S];
        var done = new float[S];
        {
            var srcObs = rollout.Observations; var srcAct = rollout.Actions;
            var srcRew = rollout.Rewards; var srcVal = rollout.Values;
            var srcDone = MemoryMarshal.Cast<byte, float>(rollout.Dones);
            for (int t = 0; t < T; t++)
                for (int n = 0; n < N; n++)
                {
                    int s = t * N + n;
                    for (int o = 0; o < O; o++) obs[s * O + o] = srcObs[(t * O + o) * N + n];
                    for (int a = 0; a < A; a++) act[s * A + a] = srcAct[(t * A + a) * N + n];
                    rew[s] = srcRew[t * N + n];
                    val[s] = srcVal[t * N + n];
                    done[s] = srcDone[t * N + n];
                }
        }

        // ---- 2. Old log-probs under the rollout policy (current weights) -------
        var oldLogp = new float[S];
        Parallel.For(0, Environment.ProcessorCount, p =>
        {
            var scratch = new Scratch(_obsDim, _actDim);
            for (int s = p; s < S; s += Environment.ProcessorCount)
            {
                Forward(obs, s, scratch);
                oldLogp[s] = LogProb(act, s, scratch.Mu);
            }
        });

        // ---- 3. GAE over transitions t = 0 .. T-2 -------------------------------
        // Engine reward timing: rew[t] is produced by the state *after* action
        // a[t-1], so transition (s_t, a_t) pairs with rew[t+1] / done[t+1].
        var adv = new float[S];
        var vTarget = new float[S];
        Parallel.For(0, N, n =>
        {
            float gae = 0f;
            for (int t = T - 2; t >= 0; t--)
            {
                int s = t * N + n, s1 = (t + 1) * N + n;
                float notDone = 1f - done[s1];
                float delta = rew[s1] + Gamma * val[s1] * notDone - val[s];
                gae = delta + Gamma * Lambda * notDone * gae;
                adv[s] = gae;
                vTarget[s] = gae + val[s];
            }
        });

        int trainS = (T - 1) * N;   // last step has no bootstrapped transition

        // Advantage normalization.
        float advMean = 0, advStd = 0;
        for (int s = 0; s < trainS; s++) advMean += adv[s];
        advMean /= trainS;
        for (int s = 0; s < trainS; s++) { float d = adv[s] - advMean; advStd += d * d; }
        advStd = MathF.Sqrt(advStd / trainS) + 1e-8f;
        for (int s = 0; s < trainS; s++) adv[s] = (adv[s] - advMean) / advStd;

        // ---- 4. Minibatch SGD epochs -------------------------------------------
        var indices = new int[trainS];
        for (int i = 0; i < trainS; i++) indices[i] = i;
        var shuffleRng = new Random(_adamT * 7919 + 13);

        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            for (int i = trainS - 1; i > 0; i--)
            {
                int j = shuffleRng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            for (int start = 0; start + MinibatchSize <= trainS; start += MinibatchSize)
                TrainMinibatch(obs, act, adv, vTarget, oldLogp, indices, start, MinibatchSize);
        }

        // ---- 5. Diagnostics ------------------------------------------------------
        float meanReward = 0; int doneCount = 0;
        for (int s = 0; s < S; s++) { meanReward += rew[s]; if (done[s] != 0) doneCount++; }
        return new UpdateResult(
            MeanReward: meanReward / S,
            MeanEpisodeLength: doneCount > 0 ? (float)S / doneCount : T);
    }

    // ------------------------------------------------------------- minibatch step

    private void TrainMinibatch(float[] obs, float[] act, float[] adv, float[] vTarget,
                                float[] oldLogp, int[] indices, int start, int count)
    {
        int workers = Math.Min(Environment.ProcessorCount, 16);
        var grads = new float[workers][];
        Parallel.For(0, workers, p =>
        {
            var g = new float[_w.Length];
            var scratch = new Scratch(_obsDim, _actDim);
            for (int i = start + p; i < start + count; i += workers)
                BackwardSample(obs, act, adv, vTarget, oldLogp, indices[i], g, scratch);
            grads[p] = g;
        });

        // Reduce thread-local gradients and take one Adam step.
        var g0 = grads[0];
        for (int p = 1; p < workers; p++)
        {
            var gp = grads[p];
            for (int k = 0; k < g0.Length; k++) g0[k] += gp[k];
        }
        float invCount = 1f / count;
        _adamT++;
        float bc1 = 1f - MathF.Pow(0.9f, _adamT), bc2 = 1f - MathF.Pow(0.999f, _adamT);
        for (int k = 0; k < _w.Length; k++)
        {
            float grad = g0[k] * invCount;
            _adamM[k] = 0.9f * _adamM[k] + 0.1f * grad;
            _adamV[k] = 0.999f * _adamV[k] + 0.001f * grad * grad;
            _w[k] -= LearningRate * (_adamM[k] / bc1) / (MathF.Sqrt(_adamV[k] / bc2) + 1e-8f);
        }
    }

    // ----------------------------------------------------------- forward/backward

    private sealed class Scratch
    {
        public readonly float[] X, Z1, H1, Z2, H2, Mu, Z3;
        public Scratch(int obsDim, int actDim)
        {
            X = new float[obsDim];
            Z1 = new float[Hidden]; H1 = new float[Hidden];
            Z2 = new float[Hidden]; H2 = new float[Hidden];
            Z3 = new float[actDim + 1];
            Mu = new float[actDim];
        }
        public float Value;
    }

    private void Forward(float[] obs, int s, Scratch sc)
    {
        Array.Copy(obs, s * _obsDim, sc.X, 0, _obsDim);
        for (int j = 0; j < Hidden; j++)
        {
            float acc = _w[_b1 + j];
            int row = _w1 + j * _obsDim;
            for (int k = 0; k < _obsDim; k++) acc += _w[row + k] * sc.X[k];
            sc.Z1[j] = acc;
            sc.H1[j] = acc > 0 ? acc : MathF.Exp(acc) - 1f;
        }
        for (int j = 0; j < Hidden; j++)
        {
            float acc = _w[_b2 + j];
            int row = _w2 + j * Hidden;
            for (int k = 0; k < Hidden; k++) acc += _w[row + k] * sc.H1[k];
            sc.Z2[j] = acc;
            sc.H2[j] = acc > 0 ? acc : MathF.Exp(acc) - 1f;
        }
        for (int j = 0; j <= _actDim; j++)
        {
            float acc = _w[_b3 + j];
            int row = _w3 + j * Hidden;
            for (int k = 0; k < Hidden; k++) acc += _w[row + k] * sc.H2[k];
            sc.Z3[j] = acc;
            if (j < _actDim) sc.Mu[j] = MathF.Tanh(acc);
        }
        sc.Value = sc.Z3[_actDim];
    }

    private float LogProb(float[] act, int s, float[] mu)
    {
        const float logSigma = -1.6094379f;          // ln(0.2)
        const float halfLog2Pi = 0.9189385f;
        float lp = 0;
        for (int a = 0; a < _actDim; a++)
        {
            float z = (act[s * _actDim + a] - mu[a]) / Sigma;
            lp += -0.5f * z * z - logSigma - halfLog2Pi;
        }
        return lp;
    }

    private void BackwardSample(float[] obs, float[] act, float[] adv, float[] vTarget,
                                float[] oldLogp, int s, float[] g, Scratch sc)
    {
        Forward(obs, s, sc);

        float newLogp = LogProb(act, s, sc.Mu);
        float ratio = MathF.Exp(Math.Clamp(newLogp - oldLogp[s], -10f, 10f));
        float advantage = adv[s];

        // Clipped surrogate: gradient flows through `ratio` only when the
        // unclipped term is active (standard PPO masking).
        bool active = advantage >= 0 ? ratio < 1f + ClipEps : ratio > 1f - ClipEps;
        float dLdLogp = active ? -advantage * ratio : 0f;

        // Value head MSE.
        float dLdV = ValueCoef * (sc.Value - vTarget[s]);

        // ---- head gradients ----
        Span<float> dZ3 = stackalloc float[_actDim + 1];
        for (int a = 0; a < _actDim; a++)
        {
            float dLdMu = dLdLogp * (act[s * _actDim + a] - sc.Mu[a]) / (Sigma * Sigma);
            dZ3[a] = dLdMu * (1f - sc.Mu[a] * sc.Mu[a]);     // tanh'
        }
        dZ3[_actDim] = dLdV;

        Span<float> dH2 = stackalloc float[Hidden];
        for (int j = 0; j <= _actDim; j++)
        {
            float dz = dZ3[j];
            if (dz == 0f) continue;
            int row = _w3 + j * Hidden;
            for (int k = 0; k < Hidden; k++)
            {
                g[row + k] += dz * sc.H2[k];
                dH2[k] += dz * _w[row + k];
            }
            g[_b3 + j] += dz;
        }

        // ---- layer 2 ----
        Span<float> dH1 = stackalloc float[Hidden];
        for (int j = 0; j < Hidden; j++)
        {
            float dz = dH2[j] * (sc.Z2[j] > 0 ? 1f : sc.H2[j] + 1f);   // ELU'
            if (dz == 0f) continue;
            int row = _w2 + j * Hidden;
            for (int k = 0; k < Hidden; k++)
            {
                g[row + k] += dz * sc.H1[k];
                dH1[k] += dz * _w[row + k];
            }
            g[_b2 + j] += dz;
        }

        // ---- layer 1 ----
        for (int j = 0; j < Hidden; j++)
        {
            float dz = dH1[j] * (sc.Z1[j] > 0 ? 1f : sc.H1[j] + 1f);
            if (dz == 0f) continue;
            int row = _w1 + j * _obsDim;
            for (int k = 0; k < _obsDim; k++) g[row + k] += dz * sc.X[k];
            g[_b1 + j] += dz;
        }
    }
}

public readonly record struct UpdateResult(float MeanReward, float MeanEpisodeLength);
