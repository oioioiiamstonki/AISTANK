# AISTANK — GPU-Resident Humanoid RL Simulation Engine

A simplified, ultra-high-performance AI simulation engine inspired by **Isaac Lab**, built for
training large batches (**4096+**) of humanoid agents to walk via reinforcement learning.

The core design goal is **maximum GPU residency**: physics state mirroring, observation
gathering, policy inference, reward computation, and termination checks all execute on the GPU
via DirectX 12 / DirectCompute, with the CPU acting only as a command-list scheduler.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ONE TICK (per frame)                         │
│                                                                     │
│  CPU: mj_step() × N envs (MuJoCo, multithreaded)                    │
│   │                                                                 │
│   ▼  (single batched upload, SoA layout)                            │
│  GPU: ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│       │ CS_Observe   │─▶│ CS_Policy    │─▶│ CS_Reward/   │          │
│       │ (obs extract)│  │ (NN forward) │  │   Terminate  │          │
│       └──────────────┘  └──────────────┘  └──────────────┘          │
│   │                                              │                  │
│   ▼  (actions readback, double-buffered)         ▼                  │
│  CPU: apply ctrl → next mj_step()         GPU: preview renderer     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🤖 Robot School — the editor

A deliberately tiny editor (think Unity/Godot reduced to its essence) designed so an
eight-year-old can use it: three giant buttons and a robot.

- **🏋️ TRAIN** — starts reinforcement learning. The skill meter and star rating climb as the
  robot learns; the cartoon robot in the viewport wobbles less and strides further.
- **▶️ PLAY** — runs the current policy with learning frozen, so you can watch what the robot
  knows so far.
- **⏹ STOP** — robot takes a break.

```sh
dotnet run --project editor
```

When the native engine is built, the viewport shows **agent 0's real MuJoCo pose** — actual
rigid-body positions read from physics each frame — and the buttons drive real GPU training
(1024 humanoids in the editor for snappy updates; the headless CLI runs the full 4096). The
skill meter tracks normalized mean reward. The window title says which mode you're in.

If the native DLL isn't built yet (or there's no DX12 GPU / MuJoCo), the editor falls back to a
built-in **demo mode** with a cartoon robot and a simulated learning curve, so the buttons
always do something.

## Layout

| Path | Contents |
|---|---|
| [`editor/`](editor) | **Robot School** — WPF editor with Train / Play / Stop buttons |
| [`src/`](src) | C++20 core engine: DX12 device/queues, MuJoCo context, simulation loop |
| [`shaders/`](shaders) | HLSL compute shaders: observation, policy inference, reward/termination |
| [`include/`](include) | Flat C ABI consumed by the C# bindings |
| [`bindings/csharp/`](bindings/csharp) | Zero-cost C# P/Invoke layer + PPO trainer stub |
| [`samples/TrainWalking/`](samples/TrainWalking) | Headless CLI training loop |
| [`assets/`](assets) | MJCF humanoid asset (16 DoF) tuned for batched walking RL |
| [`docs/`](docs) | Detailed architecture: GPU-resident loop, DX12 pipeline, policy execution |

## Pillars (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

1. **The GPU-Resident Loop** — MuJoCo steps on CPU worker threads (MuJoCo is CPU-native), but
   *everything downstream* — observation normalization, policy MLP forward pass, reward,
   termination, reset masking — runs as chained compute dispatches on a dedicated compute queue.
   Only two transfers cross PCIe per tick: one batched SoA state upload, one action readback,
   both double-buffered so the GPU never stalls.
2. **DirectX 12 Pipeline Setup** — one shader-visible CBV/SRV/UAV heap, a single shared root
   signature for all simulation compute PSOs, a separate graphics PSO for the instanced preview
   renderer, explicit fence-based lifetime management for all resources.
3. **Simplified Policy Execution** — a 2-hidden-layer MLP (obs → 256 → 256 → actions) executed
   entirely in a wave-cooperative HLSL compute shader (one threadgroup per agent, weights in
   structured buffers, activations in groupshared memory). Zero CPU-GPU sync during inference.

## Memory layout: SoA everywhere

All per-agent state lives in **Structure of Arrays** structured buffers
(`RootPosX[agent]`, `RootPosY[agent]`, …, `JointVel[dof * NUM_AGENTS + agent]`) so that a
threadgroup of 64 lanes reading "joint 3 velocity for agents 0–63" issues one fully coalesced
cache line burst instead of 64 strided AoS loads.

## Building

```sh
cmake -B build -G "Visual Studio 17 2022" -DMUJOCO_DIR=C:/path/to/mujoco-3.x
cmake --build build --config Release
dotnet run --project editor          # Robot School (works even without the native build)
dotnet run --project samples/TrainWalking   # headless training CLI
```

Requirements: Windows 10 2004+, DX12-capable GPU (SM 6.0+), [MuJoCo ≥ 3.0 C library](https://github.com/google-deepmind/mujoco/releases), CMake ≥ 3.24, .NET 8.

## Status — runs for real

This is no longer a paper scaffold. The native engine **builds and runs end-to-end**:

- **Real MuJoCo physics** — 4096 independent `mjData` humanoids stepped on a CPU worker pool
  (MuJoCo 3.9.0 C API), state mirrored into DX12 structured buffers every tick.
- **Real GPU compute** — observation gathering, the MLP policy forward pass, reward, and
  termination all run as DirectX 12 / DirectCompute dispatches (measured ~0.16 ms/tick on the
  test GPU via timestamp queries).
- **Real PPO, on the GPU** — the entire learning update (GAE(λ), advantage normalization, the
  MLP forward/backward pass, gradient reduction via a tiled GEMM, and Adam) runs as HLSL compute
  shaders in [`TrainShaders.hlsl`](shaders/TrainShaders.hlsl), orchestrated by
  [`GpuTrainer`](src/GpuTrainer.cpp), operating on the rollout buffers that **never leave VRAM**.
  The CPU [`PpoTrainer.cs`](bindings/csharp/PpoTrainer.cs) is kept as a reference implementation
  and as the oracle for a gradient-parity self-test (`TrainWalking --parity`): the GPU gradient
  matches it to a relative L2 error of ~6e-6 (cosine 1.0000). Moving the update onto the GPU made
  training ~4× faster end-to-end (CPU `Parallel.For` backprop was ~85% of iteration time);
  MuJoCo physics stepping is now the co-bottleneck.

### What runs where

| Stage | Device |
|---|---|
| MuJoCo physics (`mj_step`) | CPU (worker pool) — MuJoCo's C API is CPU-native |
| Observation gather, policy inference, reward, termination | **GPU** (DirectCompute) |
| PPO update: GAE, forward/backward, gradient reduction, Adam | **GPU** (DirectCompute) |

The only thing left on the CPU is the physics engine itself. Making *that* GPU-resident would
mean swapping MuJoCo's C API for a GPU physics backend (MJX / NVIDIA Warp) — a separate effort.

Verified on Windows 11 + an AMD/NVIDIA DX12 GPU: `dotnet run --project samples/TrainWalking`
prints live reward / steps-per-second, and **Robot School** renders agent 0's actual MuJoCo
skeleton walking (or falling) in its viewport.

### Get MuJoCo (one-time)

The engine links the prebuilt MuJoCo C library. Download a release and point CMake at it:

```sh
# https://github.com/google-deepmind/mujoco/releases  (3.9.0 tested)
# extract so that <dir>/include/mujoco/mujoco.h and <dir>/lib/mujoco.lib exist
cmake -B build -G "Visual Studio 17 2022" -A x64 -DMUJOCO_DIR=C:/path/to/mujoco-3.9.0
```
