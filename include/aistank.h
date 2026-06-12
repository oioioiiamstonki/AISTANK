// aistank.h — flat C ABI for the AISTANK engine core.
// Consumed by bindings/csharp via P/Invoke (blittable types only, no marshaling cost).
#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef AISTANK_EXPORTS
    #define AISTANK_API __declspec(dllexport)
  #else
    #define AISTANK_API __declspec(dllimport)
  #endif
#else
  #define AISTANK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AistankEngine AistankEngine;   // opaque

typedef enum AistankResult {
    AISTANK_OK              = 0,
    AISTANK_ERR_DEVICE      = 1,   // DX12 device creation / device-removed
    AISTANK_ERR_MUJOCO      = 2,   // model load / mj_step failure
    AISTANK_ERR_BAD_ARG     = 3,
    AISTANK_ERR_OOM         = 4,
} AistankResult;

typedef struct AistankConfig {
    uint32_t num_agents;        // e.g. 4096
    uint32_t physics_substeps;  // mj_step calls per Tick (e.g. 4 → 240 Hz physics @ 60 Hz control)
    uint32_t rollout_horizon;   // ticks per PPO rollout segment (e.g. 32)
    uint32_t cpu_workers;       // 0 = hardware_concurrency
    uint32_t enable_preview;    // 1 = create swapchain + render first 64 agents
    uint32_t preview_hwnd_lo;   // HWND split for blittability
    uint32_t preview_hwnd_hi;
    float    control_dt;        // seconds per control step (e.g. 1/60)
    // Reward weights (mirrored into SimConstants CBV)
    float    w_forward, w_alive, w_energy, w_upright;
    float    target_velocity;   // m/s
    float    termination_height;// root z below this → done
} AistankConfig;

typedef struct AistankStepStats {
    uint64_t tick;
    float    mean_reward;       // over agents, this tick
    float    mean_episode_len;  // over episodes terminated this tick (0 if none)
    uint32_t resets_this_tick;
    float    cpu_physics_ms;
    float    gpu_compute_ms;    // timestamp-query measured
} AistankStepStats;

// ---- Lifecycle ------------------------------------------------------------
AISTANK_API AistankResult Engine_Create(const AistankConfig* cfg,
                                        const char* mjcf_path,
                                        AistankEngine** out_engine);
AISTANK_API void          Engine_Destroy(AistankEngine* e);

// ---- Simulation -----------------------------------------------------------
// Advances physics + GPU pipeline by one control step. Non-blocking w.r.t. GPU
// except the frames-in-flight throttle.
AISTANK_API AistankResult Engine_Tick(AistankEngine* e, AistankStepStats* out_stats);

// ---- Policy weights -------------------------------------------------------
// Pushes a full FP32 parameter blob (layout documented in PolicyLayout below)
// into the inactive weight slot; takes effect next Tick. Size must match
// Engine_GetPolicyParamCount().
AISTANK_API AistankResult Engine_SetPolicyWeights(AistankEngine* e,
                                                  const float* params, uint64_t count);
AISTANK_API uint64_t      Engine_GetPolicyParamCount(const AistankEngine* e);

// ---- Rollout readback (once per horizon, NOT per tick) ---------------------
// Blocks until the in-flight horizon's readback fence completes, then exposes
// the pinned readback pointer. Layout: SoA, [horizon][component][agent].
// Valid until the next Engine_Tick. Sizes returned in out_* params.
AISTANK_API AistankResult Engine_MapRollout(AistankEngine* e,
                                            const float** out_obs,
                                            const float** out_actions,
                                            const float** out_rewards,
                                            const uint8_t** out_dones,
                                            const float** out_values,
                                            uint32_t* out_horizon,
                                            uint32_t* out_obs_dim,
                                            uint32_t* out_act_dim);

// ---- Introspection ----------------------------------------------------------
AISTANK_API uint32_t Engine_GetObservationDim(const AistankEngine* e);
AISTANK_API uint32_t Engine_GetActionDim(const AistankEngine* e);
AISTANK_API const char* Engine_GetLastError(const AistankEngine* e); // UTF-8, engine-owned

#ifdef __cplusplus
} // extern "C"
#endif
