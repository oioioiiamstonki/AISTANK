// TrainShaders.hlsl — GPU-resident PPO learning step.
//
// Moves the entire policy update (GAE → forward → backward → gradient reduction
// → Adam) onto the GPU. Operates directly on the rollout buffers the simulation
// already fills each horizon, so nothing is read back to the CPU during training.
//
// Network (identical topology to CS_PolicyForward in AgentShaders.hlsl):
//   obs[O] -> Dense(256) ELU -> Dense(256) ELU -> { mu[A]=tanh(z), V=z }
// Weight/grad layout: [W1 Hd×O][b1 Hd][W2 Hd×Hd][b2 Hd][W3 Aout×Hd][b3 Aout],
// every weight matrix row-major per output neuron (matches PpoTrainer.cs).
//
// elu'(z) is recovered from the stored activation h (h>0 <=> z>0), so the
// pre-activations never need storing: elu'(z) = (h > 0) ? 1 : (h + 1).

#define HD 256            // hidden width
#define TILE 16           // GEMM tile

cbuffer TrainConstants : register(b0)
{
    uint S;               // training samples = (H-1) * N
    uint O;               // observation dim
    uint A;               // action dim
    uint Aout;            // A + 1 (mu heads + value)
    uint N;               // agents
    uint H;               // rollout horizon (ticks)
    uint P;               // total parameter count
    uint OffW1, OffB1, OffW2, OffB2, OffW3, OffB3;
    float Gamma, Lambda, ClipEps, ValueCoef, Sigma, Lr, Beta1, Beta2, AdamEps;
    uint _pad;
};

cbuffer TrainPass : register(b1)        // root 32-bit constants, per dispatch
{
    uint Layer;           // 0,1,2 for weight-grad / bias-grad passes
    uint AdamT;           // Adam timestep (for bias correction)
    uint Flags;           // bit0: forward writes OldLogp
    uint SBase;           // sample offset (dispatches are chunked to <= 65535 groups)
};

RWStructuredBuffer<float> Weights : register(u0);   // active policy slot, updated in place
RWStructuredBuffer<float> Grad    : register(u1);
RWStructuredBuffer<float> AdamM   : register(u2);
RWStructuredBuffer<float> AdamV   : register(u3);

RWStructuredBuffer<float> RObs  : register(u4);     // rollout [(t*O+o)*N + n]
RWStructuredBuffer<float> RAct  : register(u5);     // rollout [(t*A+a)*N + n]
RWStructuredBuffer<float> RRew  : register(u6);     // rollout [t*N + n]
RWStructuredBuffer<float> RDone : register(u7);     // rollout [t*N + n]
RWStructuredBuffer<float> RVal  : register(u8);     // rollout [t*N + n]

RWStructuredBuffer<float> Adv     : register(u9);   // [S]
RWStructuredBuffer<float> VTarget : register(u10);  // [S]
RWStructuredBuffer<float> OldLogp : register(u11);  // [S]

RWStructuredBuffer<float> Xbuf : register(u12);     // [S*O] gathered obs (sample-major)
RWStructuredBuffer<float> H1   : register(u13);     // [S*HD]
RWStructuredBuffer<float> H2   : register(u14);     // [S*HD]
RWStructuredBuffer<float> dZ1  : register(u15);     // [S*HD]
RWStructuredBuffer<float> dZ2  : register(u16);     // [S*HD]
RWStructuredBuffer<float> dZ3  : register(u17);     // [S*Aout]

RWStructuredBuffer<uint>  Stats : register(u18);    // [0]=advSum [1]=advSqSum [2]=rewSum (fixed via CAS)

// ---------------------------------------------------------------- helpers

float Elu(float x)      { return x > 0.0 ? x : exp(x) - 1.0; }
float EluDFromH(float h) { return h > 0.0 ? 1.0 : h + 1.0; }   // derivative from activation

void AtomicAddF(uint idx, float val)
{
    uint cur = Stats[idx];
    uint old;
    [allow_uav_condition] do {
        old = cur;
        uint nv = asuint(asfloat(old) + val);
        InterlockedCompareExchange(Stats[idx], old, nv, cur);
    } while (cur != old);
}

float W(uint off, uint row, uint cols, uint k) { return Weights[off + row * cols + k]; }

// =============================================================================
// GAE — one thread per agent, sequential over the horizon (matches PpoTrainer).
// Also accumulates the mean-reward statistic for the UI skill meter.
// =============================================================================
[numthreads(64, 1, 1)]
void CS_GAE(uint3 dt : SV_DispatchThreadID)
{
    uint n = dt.x;
    if (n >= N) return;

    float gae = 0.0;
    [loop] for (int t = (int)H - 2; t >= 0; --t)
    {
        uint s  = (uint)t * N + n;
        uint s1 = (uint)(t + 1) * N + n;
        float notDone = 1.0 - RDone[s1];
        float delta = RRew[s1] + Gamma * RVal[s1] * notDone - RVal[s];
        gae = delta + Gamma * Lambda * notDone * gae;
        Adv[s] = gae;
        VTarget[s] = gae + RVal[s];
        AtomicAddF(0, gae);
        AtomicAddF(1, gae * gae);
    }
    // Reward stat over the whole horizon (all H ticks).
    [loop] for (uint tt = 0; tt < H; ++tt)
        AtomicAddF(2, RRew[tt * N + n]);
}

// Normalize advantages in place using the accumulated sum / sum-of-squares.
[numthreads(64, 1, 1)]
void CS_AdvNorm(uint3 dt : SV_DispatchThreadID)
{
    uint s = dt.x;
    if (s >= S) return;
    float mean = asfloat(Stats[0]) / (float)S;
    float var  = asfloat(Stats[1]) / (float)S - mean * mean;
    float std  = sqrt(max(var, 0.0)) + 1e-8;
    Adv[s] = (Adv[s] - mean) / std;
}

// =============================================================================
// Forward — one threadgroup (HD lanes) per sample. Stores H1,H2,Xbuf and,
// when Flags&1, the old log-prob under the current (pre-update) policy.
// =============================================================================
groupshared float s_x[HD];     // obs (padded)
groupshared float s_h1[HD];
groupshared float s_h2[HD];
groupshared float s_mu[64];    // action means for logp reduction (A <= 64)

[numthreads(HD, 1, 1)]
void CS_TrainForward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint sIdx = gid.x + SBase;
    if (sIdx >= S) return;
    uint j = gtid.x;
    uint t = sIdx / N, n = sIdx % N;

    // Gather this sample's observation (sample-major) for later GEMM reuse.
    if (j < O) { float xv = RObs[(t * O + j) * N + n]; s_x[j] = xv; Xbuf[sIdx * O + j] = xv; }
    else if (j < HD) s_x[j] = 0.0;
    GroupMemoryBarrierWithGroupSync();

    float acc = Weights[OffB1 + j];
    [loop] for (uint k = 0; k < O; ++k) acc += W(OffW1, j, O, k) * s_x[k];
    float h1 = Elu(acc); s_h1[j] = h1; H1[sIdx * HD + j] = h1;
    GroupMemoryBarrierWithGroupSync();

    acc = Weights[OffB2 + j];
    [loop] for (uint k2 = 0; k2 < HD; ++k2) acc += W(OffW2, j, HD, k2) * s_h1[k2];
    float h2 = Elu(acc); s_h2[j] = h2; H2[sIdx * HD + j] = h2;
    GroupMemoryBarrierWithGroupSync();

    if ((Flags & 1) && j < A)
    {
        float z = Weights[OffB3 + j];
        [loop] for (uint k3 = 0; k3 < HD; ++k3) z += W(OffW3, j, HD, k3) * s_h2[k3];
        s_mu[j] = tanh(z);
    }
    GroupMemoryBarrierWithGroupSync();

    if ((Flags & 1) && j == 0)
    {
        const float logSig = log(Sigma), halfLog2Pi = 0.9189385;
        float lp = 0.0;
        [loop] for (uint a = 0; a < A; ++a)
        {
            float zz = (RAct[(t * A + a) * N + n] - s_mu[a]) / Sigma;
            lp += -0.5 * zz * zz - logSig - halfLog2Pi;
        }
        OldLogp[sIdx] = lp;
    }
}

// =============================================================================
// Backward — one threadgroup (HD lanes) per sample. Reads H1,H2; recomputes the
// heads; produces dZ1,dZ2,dZ3 (the per-sample pre-activation gradients).
// =============================================================================
groupshared float sb_h1[HD];
groupshared float sb_h2[HD];
groupshared float sb_dz2[HD];
groupshared float sb_dz3[64];
groupshared float sb_mu[64];
groupshared float sb_newlogp;

[numthreads(HD, 1, 1)]
void CS_TrainBackward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint sIdx = gid.x + SBase;
    if (sIdx >= S) return;
    uint j = gtid.x;
    uint t = sIdx / N, n = sIdx % N;

    float h1 = H1[sIdx * HD + j]; sb_h1[j] = h1;
    float h2 = H2[sIdx * HD + j]; sb_h2[j] = h2;
    GroupMemoryBarrierWithGroupSync();

    // Heads: z3[j] for j < Aout (mu = tanh, value = z3[A]).
    float z3 = 0.0;
    if (j < Aout)
    {
        z3 = Weights[OffB3 + j];
        [loop] for (uint k = 0; k < HD; ++k) z3 += W(OffW3, j, HD, k) * sb_h2[k];
        if (j < A) sb_mu[j] = tanh(z3);
    }
    GroupMemoryBarrierWithGroupSync();

    if (j == 0)
    {
        const float logSig = log(Sigma), halfLog2Pi = 0.9189385;
        float lp = 0.0;
        [loop] for (uint a = 0; a < A; ++a)
        {
            float zz = (RAct[(t * A + a) * N + n] - sb_mu[a]) / Sigma;
            lp += -0.5 * zz * zz - logSig - halfLog2Pi;
        }
        sb_newlogp = lp;
    }
    GroupMemoryBarrierWithGroupSync();

    // dZ3: policy (clipped surrogate) for action heads, value MSE for the last.
    if (j < Aout)
    {
        float adv = Adv[sIdx];
        float ratio = exp(clamp(sb_newlogp - OldLogp[sIdx], -10.0, 10.0));
        bool active = adv >= 0.0 ? ratio < 1.0 + ClipEps : ratio > 1.0 - ClipEps;
        float dLogp = active ? -adv * ratio : 0.0;
        float g;
        if (j < A)
        {
            float mu = sb_mu[j];
            float act = RAct[(t * A + j) * N + n];
            float dMu = dLogp * (act - mu) / (Sigma * Sigma);
            g = dMu * (1.0 - mu * mu);              // tanh'
        }
        else
        {
            float value = z3;                        // head A = value
            g = ValueCoef * (value - VTarget[sIdx]);
        }
        sb_dz3[j] = g;
        dZ3[sIdx * Aout + j] = g;
    }
    GroupMemoryBarrierWithGroupSync();

    // dH2 = dZ3 · W3 ; dZ2 = dH2 * elu'(z2)
    float dh2 = 0.0;
    [loop] for (uint m = 0; m < Aout; ++m) dh2 += sb_dz3[m] * W(OffW3, m, HD, j);
    float dz2 = dh2 * EluDFromH(sb_h2[j]);
    sb_dz2[j] = dz2; dZ2[sIdx * HD + j] = dz2;
    GroupMemoryBarrierWithGroupSync();

    // dH1 = dZ2 · W2 ; dZ1 = dH1 * elu'(z1)
    float dh1 = 0.0;
    [loop] for (uint m2 = 0; m2 < HD; ++m2) dh1 += sb_dz2[m2] * W(OffW2, m2, HD, j);
    float dz1 = dh1 * EluDFromH(sb_h1[j]);
    dZ1[sIdx * HD + j] = dz1;
}

// =============================================================================
// Weight-gradient reduction: dW[m,k] = sum_s dZ[s,m] * inp[s,k]  (tiled A^T·B).
// Layer selects (dZ, inp, M, K). One dispatch per layer.
// =============================================================================
groupshared float tA[TILE][TILE];   // dZ tile  (A^T view)
groupshared float tB[TILE][TILE];   // inp tile

void LayerDims(out uint M, out uint K, out uint woff)
{
    if (Layer == 0)      { M = HD;   K = O;  woff = OffW1; }
    else if (Layer == 1) { M = HD;   K = HD; woff = OffW2; }
    else                 { M = Aout; K = HD; woff = OffW3; }
}
float DZ(uint s, uint m)
{
    if (Layer == 0) return dZ1[s * HD + m];
    if (Layer == 1) return dZ2[s * HD + m];
    return dZ3[s * Aout + m];
}
float INP(uint s, uint k)
{
    if (Layer == 0) return Xbuf[s * O + k];
    if (Layer == 1) return H1[s * HD + k];
    return H2[s * HD + k];
}

[numthreads(TILE, TILE, 1)]
void CS_WeightGrad(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint M, K, woff; LayerDims(M, K, woff);
    uint m = gid.x * TILE + gtid.y;   // output row (out neuron)
    uint k = gid.y * TILE + gtid.x;   // output col (input index)

    float acc = 0.0;
    uint tiles = (S + TILE - 1) / TILE;
    for (uint tIdx = 0; tIdx < tiles; ++tIdx)
    {
        uint sA = tIdx * TILE + gtid.x;        // sample for tA[m]
        uint sB = tIdx * TILE + gtid.y;        // sample for tB[k]
        tA[gtid.y][gtid.x] = (m < M && sA < S) ? DZ(sA, m) : 0.0;
        tB[gtid.y][gtid.x] = (k < K && sB < S) ? INP(sB, k) : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint i = 0; i < TILE; ++i) acc += tA[gtid.y][i] * tB[i][gtid.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (m < M && k < K) Grad[woff + m * K + k] = acc;
}

// Bias gradient: db[m] = sum_s dZ[s,m]. One thread per output neuron.
[numthreads(64, 1, 1)]
void CS_BiasGrad(uint3 dt : SV_DispatchThreadID)
{
    uint M, K, woff; LayerDims(M, K, woff);
    uint boff = (Layer == 0) ? OffB1 : (Layer == 1) ? OffB2 : OffB3;
    uint m = dt.x;
    if (m >= M) return;
    float acc = 0.0;
    [loop] for (uint s = 0; s < S; ++s) acc += DZ(s, m);
    Grad[boff + m] = acc;
}

// =============================================================================
// Adam — one thread per parameter. Gradients are summed over S samples, so we
// scale by 1/S to match the CPU trainer's per-sample mean.
// =============================================================================
[numthreads(256, 1, 1)]
void CS_Adam(uint3 dt : SV_DispatchThreadID)
{
    uint i = dt.x;
    if (i >= P) return;
    float g = Grad[i] / (float)S;
    float m = Beta1 * AdamM[i] + (1.0 - Beta1) * g;
    float v = Beta2 * AdamV[i] + (1.0 - Beta2) * g * g;
    AdamM[i] = m; AdamV[i] = v;
    float bc1 = 1.0 - pow(Beta1, (float)AdamT);
    float bc2 = 1.0 - pow(Beta2, (float)AdamT);
    Weights[i] -= Lr * (m / bc1) / (sqrt(v / bc2) + AdamEps);
}

// Zero the stats buffer (3 entries) before a GAE accumulation pass.
[numthreads(8, 1, 1)]
void CS_ZeroStats(uint3 dt : SV_DispatchThreadID)
{
    if (dt.x < 8) Stats[dt.x] = 0;
}
