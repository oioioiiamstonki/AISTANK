// AgentShaders.hlsl — GPU-resident RL kernels for batched humanoid locomotion.
//
// All per-agent state is SoA: flat float buffers indexed [component * NumAgents + agent]
// so each 64-lane threadgroup issues fully coalesced loads.
//
// Entry points (compiled separately by CMake/DXC, cs_6_0):
//   CS_GatherObservations   — normalize physics state into the policy input vector
//   CS_PolicyForward        — MLP inference, one 256-lane threadgroup per agent
//   CS_RewardAndTerminate   — forward-velocity reward, fall detection, rollout append
//   CS_ApplyResets          — zero per-agent episode state for done agents
//   CS_BuildInstances       — pack preview transforms for the renderer (first 64 agents)

// ---------------------------------------------------------------- Root bindings
// Must match EngineCore::CreateRootSignatures and GpuBuffers.h slot enums.

cbuffer SimConstants : register(b0)
{
    uint  NumAgents;
    uint  DofCount;            // actuated joints (e.g. 16)
    uint  ActionDim;
    uint  ObsDim;
    float ControlDt;
    float TargetVelocity;      // m/s, forward reward saturates here
    float TerminationHeight;   // root z below this => fallen
    float WForward;
    float WAlive;
    float WEnergy;
    float WUpright;
    uint  MaxEpisodeSteps;
    uint  RolloutHorizon;
    uint  RolloutCursor;       // tick % horizon
    uint  _pad0, _pad1;
};

cbuffer PassConstants : register(b1)   // 4 root constants
{
    uint PassIndex;
    uint LayerIndex;           // unused by the fused MLP kernel
    uint WeightSlot;           // 0/1, double-buffered policy params
    uint RngSeed;
};

// Physics mirror (SoA, written by the copy queue each tick)
StructuredBuffer<float> JointPos     : register(t0);   // [DofCount * N]
StructuredBuffer<float> JointVel     : register(t1);   // [DofCount * N]
StructuredBuffer<float> RootPos      : register(t2);   // [3 * N] (x,y,z planes)
StructuredBuffer<float> RootQuat     : register(t3);   // [4 * N] (w,x,y,z — MuJoCo order)
StructuredBuffer<float> RootLinVel   : register(t4);   // [3 * N]
StructuredBuffer<float> RootAngVel   : register(t5);   // [3 * N]
StructuredBuffer<float> InitialState : register(t6);

StructuredBuffer<float> Weights0     : register(t32);  // policy params, slot 0
StructuredBuffer<float> Weights1     : register(t33);  // policy params, slot 1

RWStructuredBuffer<float> Observations : register(u0); // [ObsDim * N]
RWStructuredBuffer<float> Actions      : register(u1); // [ActionDim * N]
RWStructuredBuffer<float> Rewards      : register(u2); // [N]
RWStructuredBuffer<uint>  DoneFlags    : register(u3); // [N]
RWStructuredBuffer<uint>  StepCount    : register(u4); // [N]
RWStructuredBuffer<uint>  RngState     : register(u5); // [N] xorshift state
RWStructuredBuffer<float> RolloutObs   : register(u6); // [H * ObsDim * N]
RWStructuredBuffer<float> RolloutAct   : register(u7); // [H * ActionDim * N]
RWStructuredBuffer<float> RolloutRew   : register(u8); // [H * N]
RWStructuredBuffer<float> RolloutDone  : register(u9); // [H * N]
RWStructuredBuffer<float> RolloutVal   : register(u10);// [H * N]
RWStructuredBuffer<uint>  EpisodeStats : register(u11);// [4] atomics
RWStructuredBuffer<float> InstanceXf   : register(u12);// [64 * 16] preview float4x4

// --------------------------------------------------------------------- Helpers

float3 QuatRotate(float4 q /*w,x,y,z*/, float3 v)
{
    float3 u = q.yzw;
    return v + 2.0 * cross(u, cross(u, v) + q.x * v);
}

uint XorShift(inout uint s)
{
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

float RandUniform(inout uint s) { return float(XorShift(s)) * 2.3283064e-10; }

// =============================================================================
// PASS 1 — Observation gathering. 64 agents per group, coalesced SoA reads.
// Obs layout per agent: [jointPos*D | jointVel*D | rootZ | rootQuat*4 |
//                        localLinVel*3 | localAngVel*3 | prevActions*A]
// =============================================================================
[numthreads(64, 1, 1)]
void CS_GatherObservations(uint3 dtid : SV_DispatchThreadID)
{
    const uint agent = dtid.x;
    if (agent >= NumAgents) return;
    const uint N = NumAgents;
    uint o = 0;

    [loop] for (uint d = 0; d < DofCount; ++d)
        Observations[(o++) * N + agent] = JointPos[d * N + agent];
    [loop] for (uint d2 = 0; d2 < DofCount; ++d2)
        Observations[(o++) * N + agent] = JointVel[d2 * N + agent] * 0.1;  // velocity scale

    Observations[(o++) * N + agent] = RootPos[2 * N + agent];              // height

    float4 q = float4(RootQuat[0 * N + agent], RootQuat[1 * N + agent],
                      RootQuat[2 * N + agent], RootQuat[3 * N + agent]);
    Observations[(o++) * N + agent] = q.x;
    Observations[(o++) * N + agent] = q.y;
    Observations[(o++) * N + agent] = q.z;
    Observations[(o++) * N + agent] = q.w;

    // World velocities → torso-local frame (yaw-invariant policy input).
    float4 qInv = float4(q.x, -q.yzw);
    float3 linW = float3(RootLinVel[0 * N + agent], RootLinVel[1 * N + agent],
                         RootLinVel[2 * N + agent]);
    float3 angW = float3(RootAngVel[0 * N + agent], RootAngVel[1 * N + agent],
                         RootAngVel[2 * N + agent]);
    float3 linL = QuatRotate(qInv, linW);
    float3 angL = QuatRotate(qInv, angW);
    Observations[(o++) * N + agent] = linL.x;
    Observations[(o++) * N + agent] = linL.y;
    Observations[(o++) * N + agent] = linL.z;
    Observations[(o++) * N + agent] = angL.x;
    Observations[(o++) * N + agent] = angL.y;
    Observations[(o++) * N + agent] = angL.z;

    [loop] for (uint a = 0; a < ActionDim; ++a)
        Observations[(o++) * N + agent] = Actions[a * N + agent];          // prev actions
}

// =============================================================================
// PASS 2 — Policy inference. One threadgroup per agent (Dispatch(N,1,1)).
// MLP: obs → 256 ELU → 256 ELU → { mu[ActionDim] (tanh), value[1] }.
// Weights row-major per output neuron:
//   [W1: 256×ObsDim][b1: 256][W2: 256×256][b2: 256][W3: (A+1)×256][b3: A+1]
// Thread j owns hidden neuron j; activations live in groupshared.
// =============================================================================
#define HIDDEN 256

groupshared float s_in[HIDDEN];     // current layer input (obs padded, or prev hidden)
groupshared float s_out[HIDDEN];

float LoadW(uint idx) { return (WeightSlot == 0) ? Weights0[idx] : Weights1[idx]; }
float Elu(float x)    { return x > 0.0 ? x : exp(x) - 1.0; }

[numthreads(HIDDEN, 1, 1)]
void CS_PolicyForward(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint agent = gid.x;
    const uint j = gtid.x;
    const uint N = NumAgents;

    // Cooperative load of this agent's observation vector (ObsDim <= HIDDEN).
    s_in[j] = (j < ObsDim) ? Observations[j * N + agent] : 0.0;
    GroupMemoryBarrierWithGroupSync();

    // ---- Layer 1: HIDDEN × ObsDim ----
    uint base = 0;
    {
        float acc = 0.0;
        uint row = base + j * ObsDim;
        [loop] for (uint k = 0; k < ObsDim; ++k)
            acc += LoadW(row + k) * s_in[k];                // s_in broadcast: no bank conflicts
        acc += LoadW(base + HIDDEN * ObsDim + j);           // b1
        s_out[j] = Elu(acc);
    }
    base += HIDDEN * ObsDim + HIDDEN;
    GroupMemoryBarrierWithGroupSync();
    s_in[j] = s_out[j];
    GroupMemoryBarrierWithGroupSync();

    // ---- Layer 2: HIDDEN × HIDDEN ----
    {
        float acc = 0.0;
        uint row = base + j * HIDDEN;
        [unroll(8)] for (uint k = 0; k < HIDDEN; ++k)
            acc += LoadW(row + k) * s_in[k];
        acc += LoadW(base + HIDDEN * HIDDEN + j);           // b2
        s_out[j] = Elu(acc);
    }
    base += HIDDEN * HIDDEN + HIDDEN;
    GroupMemoryBarrierWithGroupSync();
    s_in[j] = s_out[j];
    GroupMemoryBarrierWithGroupSync();

    // ---- Heads: rows 0..A-1 = action mu (tanh), row A = value ----
    const uint headRows = ActionDim + 1;
    if (j < headRows)
    {
        float acc = 0.0;
        uint row = base + j * HIDDEN;
        [unroll(8)] for (uint k = 0; k < HIDDEN; ++k)
            acc += LoadW(row + k) * s_in[k];
        acc += LoadW(base + headRows * HIDDEN + j);         // b3

        if (j < ActionDim)
        {
            // Exploration noise: per-agent xorshift, fixed log-std (matches trainer).
            uint rng = RngState[agent] ^ (RngSeed * 747796405u + j * 2891336453u);
            float u1 = max(RandUniform(rng), 1e-7);
            float u2 = RandUniform(rng);
            float gauss = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
            RngState[agent] = rng;
            Actions[j * N + agent] = clamp(tanh(acc) + 0.2 * gauss, -1.0, 1.0);
        }
        else
        {
            RolloutVal[RolloutCursor * N + agent] = acc;    // critic value estimate
        }
    }
}

// =============================================================================
// PASS 3 — Reward + termination + rollout append. 64 agents per group.
// =============================================================================
[numthreads(64, 1, 1)]
void CS_RewardAndTerminate(uint3 dtid : SV_DispatchThreadID)
{
    const uint agent = dtid.x;
    if (agent >= NumAgents) return;
    const uint N = NumAgents;

    // --- Forward velocity (world x is the walk direction) ---
    float vx = RootLinVel[0 * N + agent];
    float rForward = WForward * clamp(vx, 0.0, TargetVelocity);

    // --- Energy penalty: Σ |action_i * joint_vel_i| (torque proxy × velocity) ---
    float energy = 0.0;
    [loop] for (uint a = 0; a < ActionDim; ++a)
        energy += abs(Actions[a * N + agent] * JointVel[a * N + agent]);

    // --- Uprightness: torso +Z axis vs world up ---
    float4 q = float4(RootQuat[0 * N + agent], RootQuat[1 * N + agent],
                      RootQuat[2 * N + agent], RootQuat[3 * N + agent]);
    float3 torsoUp = QuatRotate(q, float3(0, 0, 1));
    float upright = torsoUp.z;                              // 1 = vertical, <0 = inverted

    float reward = rForward
                 + WAlive
                 - WEnergy * energy
                 - WUpright * (1.0 - upright);

    // --- Termination: fell, tipped over, or episode timeout ---
    float rootZ = RootPos[2 * N + agent];
    uint steps = StepCount[agent] + 1;
    bool fell    = rootZ < TerminationHeight;
    bool tipped  = upright < 0.3;                           // ~72° from vertical
    bool timeout = steps >= MaxEpisodeSteps;
    uint done = (fell || tipped) ? 1u : (timeout ? 2u : 0u); // 2 = truncation (no fail penalty)

    if (done == 1u) reward -= 1.0;                           // terminal fall penalty

    Rewards[agent]   = reward;
    DoneFlags[agent] = done;
    StepCount[agent] = steps;

    // --- Rollout append (SoA: [cursor][component][agent]) ---
    const uint c = RolloutCursor;
    [loop] for (uint o = 0; o < ObsDim; ++o)
        RolloutObs[(c * ObsDim + o) * N + agent] = Observations[o * N + agent];
    [loop] for (uint a2 = 0; a2 < ActionDim; ++a2)
        RolloutAct[(c * ActionDim + a2) * N + agent] = Actions[a2 * N + agent];
    RolloutRew [c * N + agent] = reward;
    RolloutDone[c * N + agent] = (done != 0u) ? 1.0 : 0.0;

    // --- Aggregate stats (fixed-point atomics: reward scaled by 1024) ---
    InterlockedAdd(EpisodeStats[0], uint(max(reward, -32.0) * 1024.0 + 32768.0));
    if (done != 0u)
    {
        InterlockedAdd(EpisodeStats[1], steps);
        InterlockedAdd(EpisodeStats[2], 1u);
    }
}

// =============================================================================
// PASS 4 — Reset bookkeeping for done agents (CPU does the mj_resetData; this
// kernel zeroes GPU-side episode state so tick N+1 starts clean).
// =============================================================================
[numthreads(64, 1, 1)]
void CS_ApplyResets(uint3 dtid : SV_DispatchThreadID)
{
    const uint agent = dtid.x;
    if (agent >= NumAgents) return;

    if (DoneFlags[agent] != 0u)
    {
        StepCount[agent] = 0;
        RngState[agent]  = agent * 1664525u + RngSeed * 1013904223u + 1u;
        // Previous-action slice of the next observation must read zero:
        [loop] for (uint a = 0; a < ActionDim; ++a)
            Actions[a * NumAgents + agent] = 0.0;
    }
}

// =============================================================================
// PASS 5 — Preview instance transforms (first 64 agents). The renderer consumes
// InstanceXf directly — sim state never returns to the CPU for visualization.
// =============================================================================
[numthreads(64, 1, 1)]
void CS_BuildInstances(uint3 dtid : SV_DispatchThreadID)
{
    const uint agent = dtid.x;
    if (agent >= min(NumAgents, 64u)) return;
    const uint N = NumAgents;

    float4 q = float4(RootQuat[0 * N + agent], RootQuat[1 * N + agent],
                      RootQuat[2 * N + agent], RootQuat[3 * N + agent]);
    float3 p = float3(RootPos[0 * N + agent], RootPos[1 * N + agent],
                      RootPos[2 * N + agent]);
    // Lay agents out on a preview grid so they don't overlap visually.
    p.y += float(agent % 8) * 2.0;
    p.x += float(agent / 8) * 2.0 - float(agent) * 0.0;

    float3 r0 = QuatRotate(q, float3(1, 0, 0));
    float3 r1 = QuatRotate(q, float3(0, 1, 0));
    float3 r2 = QuatRotate(q, float3(0, 0, 1));

    const uint b = agent * 16;
    InstanceXf[b +  0] = r0.x; InstanceXf[b +  1] = r1.x; InstanceXf[b +  2] = r2.x; InstanceXf[b +  3] = p.x;
    InstanceXf[b +  4] = r0.y; InstanceXf[b +  5] = r1.y; InstanceXf[b +  6] = r2.y; InstanceXf[b +  7] = p.y;
    InstanceXf[b +  8] = r0.z; InstanceXf[b +  9] = r1.z; InstanceXf[b + 10] = r2.z; InstanceXf[b + 11] = p.z;
    InstanceXf[b + 12] = 0.0;  InstanceXf[b + 13] = 0.0;  InstanceXf[b + 14] = 0.0;  InstanceXf[b + 15] = 1.0;
}
