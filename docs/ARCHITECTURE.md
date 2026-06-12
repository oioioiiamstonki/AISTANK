# AISTANK Architecture

Target: 4096+ simplified humanoids (16 DoF) learning to walk, with the entire RL inner loop
(observe → infer → act → reward → terminate → reset) resident on the GPU.

---

## 1. The GPU-Resident Loop

### 1.1 The honest constraint: MuJoCo steps on the CPU

MuJoCo's C API (`mj_step`) is CPU-native. Rather than pretend otherwise, the engine treats
MuJoCo as a *producer* feeding a GPU-resident consumer pipeline, and engineers the boundary so
it costs exactly **one batched upload and one batched readback per tick**:

- N independent `mjData` instances are stepped by a work-stealing thread pool
  (one `mjModel`, shared read-only; `mjData` is per-env, so stepping is embarrassingly parallel).
- After stepping, each worker scatters its env's `qpos/qvel/cfrc_ext` slices into a **pinned,
  SoA-laid-out upload staging buffer** (`D3D12_HEAP_TYPE_UPLOAD`, persistently mapped).
- One `CopyBufferRegion` on the **copy queue** moves the whole batch to default-heap
  structured buffers. The copy queue overlaps with the previous tick's compute work.

Everything after that copy is GPU-only:

```
tick N timeline (three queues, fence-synchronized):

COPY    : [upload state N]                    [readback actions N]
COMPUTE :        [Observe N][Policy N][Reward/Term N][ResetMask N]
DIRECT  :                                  [preview render N-1 (optional)]
CPU     : [mj_step N+1 on workers, overlapped with all of the above]
```

The CPU applies actions from tick **N-1** while the GPU computes actions for tick **N**
(standard one-step-lag pipelining; the policy is trained with this latency, so it is not a
correctness issue, and it buys full CPU/GPU overlap).

### 1.2 Per-tick dispatch chain (all on the compute queue)

| Pass | Kernel | Dispatch shape | Reads | Writes |
|---|---|---|---|---|
| 1 | `CS_GatherObservations` | `ceil(N/64), 1, 1` | qpos, qvel SoA | Observations (normalized) |
| 2 | `CS_PolicyForward` | `N, 1, 1` (1 group/agent) | Observations, Weights | Actions, ValueEst |
| 3 | `CS_RewardAndTerminate` | `ceil(N/64), 1, 1` | qpos, qvel, Actions | Rewards, DoneFlags, EpisodeStats |
| 4 | `CS_ApplyResets` | `ceil(N/64), 1, 1` | DoneFlags, InitialState | qpos/qvel reset slots, RNG states |

Passes are separated by UAV barriers only — no fence round-trips, no PSO heap switches
(everything shares one root signature and one descriptor heap).

### 1.3 SoA layout

For `N` agents, `D` DoF, all buffers are flat `StructuredBuffer<float>` indexed
`[component * N + agent]`:

```
JointPos   : float[D * N]      JointVel   : float[D * N]
RootPos    : float[3 * N]      RootQuat   : float[4 * N]
RootLinVel : float[3 * N]      RootAngVel : float[3 * N]
Actions    : float[A * N]      Rewards    : float[N]
DoneFlags  : uint[N]           StepCount  : uint[N]
```

Lane `i` of a 64-wide threadgroup processing agents `base..base+63` reads
`JointVel[d * N + base + i]` — consecutive addresses, perfectly coalesced. The AoS
alternative (`struct Agent { float qpos[16]; ... }`) would waste ~94% of each cache line per
component access.

---

## 2. DirectX 12 Pipeline Setup

### 2.1 Queues

| Queue | Type | Role |
|---|---|---|
| `m_computeQueue` | `D3D12_COMMAND_LIST_TYPE_COMPUTE` | All simulation kernels |
| `m_copyQueue` | `D3D12_COMMAND_LIST_TYPE_COPY` | State upload / action readback (overlaps compute) |
| `m_directQueue` | `D3D12_COMMAND_LIST_TYPE_DIRECT` | Preview renderer + swapchain present |

One `ID3D12Fence` per queue; cross-queue dependencies expressed with `Wait`/`Signal`, never CPU
waits inside the tick (the only CPU wait is the frames-in-flight throttle).

### 2.2 Descriptor heaps

- **One** shader-visible `CBV_SRV_UAV` heap (4096 descriptors), set once per command list.
  Layout is a fixed table: slots `[0..15]` simulation SRVs, `[16..31]` simulation UAVs,
  `[32..47]` policy weight SRVs, `[48..]` renderer SRVs. Fixed slots → descriptors are written
  once at init, never churned per frame.
- One non-shader-visible RTV heap (swapchain backbuffers ×3) and DSV heap (depth ×1) for the
  preview renderer.
- Sampler heap: none needed for sim; renderer uses static samplers in its root signature.

### 2.3 Root signatures

**Compute (shared by all 4 sim PSOs + policy):**

```
[0] CBV  b0  : SimConstants (N, dt, reward scales, layer dims) — root CBV, no table
[1] TABLE    : SRV t0-t15 (state + weights)
[2] TABLE    : UAV u0-u15 (obs, actions, rewards, dones, RNG)
[3] 32BIT b1 : 4 root constants (passIndex, layerIndex, rngSeed, flags)
```

Root constants let one command list re-dispatch the same policy PSO per layer without a CBV
update. Flag: `D3D12_ROOT_SIGNATURE_FLAG_NONE` (no IA needed for compute).

**Graphics (preview):** root CBV (camera), SRV table (instance transforms — written by a small
`CS_BuildInstances` pass directly from sim state, so rendering never reads back to CPU),
`ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`.

### 2.4 PSOs

| PSO | Shader | Notes |
|---|---|---|
| `m_psoObserve` | `CS_GatherObservations` | cs_6_0, 64-lane groups |
| `m_psoPolicy` | `CS_PolicyForward` | cs_6_0, 256-lane groups, groupshared activations |
| `m_psoReward` | `CS_RewardAndTerminate` | cs_6_0 |
| `m_psoReset` | `CS_ApplyResets` | cs_6_0 |
| `m_psoPreviewGfx` | VS/PS instanced capsules | One draw, `N` instances, depth-only lighting |

All compute PSOs compiled offline (DXC) to `.cso`; cached `ID3D12PipelineLibrary` optional.

### 2.5 Resource lifetimes (explicit rules)

1. **Default-heap sim buffers**: created once at `Engine_Create`, released at `Engine_Destroy`.
   Never recreated mid-run; capacity is fixed by `maxAgents`.
2. **Upload/readback rings**: 2 frames in flight × (state staging + action readback). A slot is
   reused only after its fence value is reached (`m_frameFence->GetCompletedValue()` check; CPU
   `WaitForSingleObject` only if the ring is full — the throttle point).
3. **Transient barriers**: UAV barriers between passes; `COPY_DEST → NON_PIXEL_SHADER_RESOURCE`
   transitions recorded on the copy list's *receiving* queue via split barriers where profitable.
4. **Weights**: uploaded through the same ring when the trainer pushes new parameters
   (every PPO iteration, not every tick); double-buffered so inference never reads a buffer
   being written (`weightsSlot = iteration & 1`, root constant selects the slot).
5. All COM lifetimes via `Microsoft::WRL::ComPtr`; no raw `Release` calls; device-removed
   handling funnels through `CheckDeviceRemoved()` after every `ExecuteCommandLists`.

---

## 3. Simplified Policy Execution (GPU-side inference)

### 3.1 Strategy

A walking policy does not need cuDNN. The actor-critic here is a small MLP:

```
obs[45] → Dense(256) + ELU → Dense(256) + ELU → { mu[16], value[1] }
```

That is ~78K parameters ≈ 312 KB FP32 — it fits in L2 of any modern GPU and is served from
structured buffers at full bandwidth. We run it with a **custom HLSL kernel, one threadgroup
per agent**:

- 256 threads per group = one thread per hidden neuron.
- Input observations are cooperatively loaded into `groupshared float s_act[256]` once.
- Each layer: thread `j` computes `dot(W[j], s_act) + b[j]`, barrier, activation, write back to
  groupshared. Weights are read coalesced because `W` is stored **row-major per output neuron**
  and all threads stream the same `s_act` (broadcast from groupshared — no bank conflicts).
- Final layer writes `Actions[a * N + agent]` (SoA) and `ValueEst[agent]` directly to UAVs.

4096 agents → 4096 groups → saturates any GPU ≥ 30 SMs; measured-style estimate: the whole
forward pass is < 200 µs on a mid-range card, fully overlapped with CPU physics.

**Why not a tensor library?** Direct binding to e.g. DirectML or onnxruntime-dml is the upgrade
path (the `IPolicyExecutor` seam in `SimulationLoop.h` exists for exactly that), but for an MLP
this small a hand-rolled kernel avoids per-dispatch graph overhead, extra descriptor churn, and
any intermediate tensors in default heap — and it keeps the entire tick inside one command list.

### 3.2 Training split

- **Inference + rollout storage**: 100% GPU. Rollout buffers (obs/act/logp/reward/done/value ×
  horizon) are default-heap UAVs appended by `CS_RewardAndTerminate`.
- **PPO update**: the scaffold reads rollouts back once per *horizon* (e.g. every 32 ticks, not
  every tick — amortized to noise) into the C# trainer, which owns the optimizer. Updated
  weights go back through the weight ring. A future `CS_AdamStep` kernel can delete even that.

### 3.3 Reward function (walking)

```
r = w_fwd  * clamp(v_x, 0, v_target)            // forward progress
  + w_alive * 1                                  // survival bonus
  - w_energy * Σ |action_i * joint_vel_i|        // energy / torque penalty
  - w_upright * (1 - dot(torso_up, world_up))    // posture
terminate if root_z < 0.55 m  OR  pitch/roll > 1.0 rad  OR  step > 1000
```

All computed in `CS_RewardAndTerminate` (see [shaders/AgentShaders.hlsl](../shaders/AgentShaders.hlsl)).
