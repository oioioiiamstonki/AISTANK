// AistankApi.cpp — flat C ABI implementation bridging to EngineCore/SimulationLoop.
#include "../include/aistank.h"
#include "EngineCore.h"
#include "SimulationLoop.h"

#include <memory>
#include <string>

// Opaque handle: owns core + loop. Loop declared after core so it is destroyed
// first (it references core's queues during teardown).
struct AistankEngine {
    std::unique_ptr<aistank::EngineCore>     core;
    std::unique_ptr<aistank::SimulationLoop> loop;
    std::string lastError;
};

namespace aistank { void SetPolicyParamBytesForUpload(uint64_t); }

extern "C" {

AISTANK_API AistankResult Engine_Create(const AistankConfig* cfg,
                                        const char* mjcf_path,
                                        AistankEngine** out_engine) {
    if (!cfg || !mjcf_path || !out_engine) return AISTANK_ERR_BAD_ARG;
    auto* e = new (std::nothrow) AistankEngine();
    if (!e) return AISTANK_ERR_OOM;
    try {
        e->core = std::make_unique<aistank::EngineCore>(*cfg, mjcf_path);
        aistank::SetPolicyParamBytesForUpload(e->core->PolicyParamCount() * 4);
        e->loop = std::make_unique<aistank::SimulationLoop>(*e->core);
        *out_engine = e;
        return AISTANK_OK;
    } catch (const std::exception& ex) {
        e->lastError = ex.what();
        // Leak-free failure path: report the error via a static slot since the
        // handle is being destroyed. Callers get the code; message goes to stderr.
        fprintf(stderr, "[aistank] Engine_Create failed: %s\n", ex.what());
        delete e;
        *out_engine = nullptr;
        return AISTANK_ERR_DEVICE;
    }
}

AISTANK_API void Engine_Destroy(AistankEngine* e) {
    delete e;  // loop first (member order), then core (drains queues)
}

AISTANK_API AistankResult Engine_Tick(AistankEngine* e, AistankStepStats* out_stats) {
    if (!e || !out_stats) return AISTANK_ERR_BAD_ARG;
    return e->loop->Tick(*out_stats);
}

AISTANK_API AistankResult Engine_SetPolicyWeights(AistankEngine* e,
                                                  const float* params, uint64_t count) {
    if (!e || !params) return AISTANK_ERR_BAD_ARG;
    return e->loop->SetPolicyWeights(params, count);
}

AISTANK_API uint64_t Engine_GetPolicyParamCount(const AistankEngine* e) {
    return e ? e->core->PolicyParamCount() : 0;
}

AISTANK_API AistankResult Engine_MapRollout(AistankEngine* e,
                                            const float** obs, const float** act,
                                            const float** rew, const uint8_t** done,
                                            const float** val,
                                            uint32_t* horizon, uint32_t* obsDim,
                                            uint32_t* actDim) {
    if (!e) return AISTANK_ERR_BAD_ARG;
    return e->loop->MapRollout(obs, act, rew, done, val, horizon, obsDim, actDim);
}

AISTANK_API uint32_t Engine_GetBodyCount(const AistankEngine* e) {
    return e ? static_cast<uint32_t>(e->core->Model()->nbody) : 0;
}

AISTANK_API AistankResult Engine_GetBodyParents(const AistankEngine* e, int32_t* out_parents) {
    if (!e || !out_parents) return AISTANK_ERR_BAD_ARG;
    const mjModel* m = e->core->Model();
    for (int b = 0; b < m->nbody; ++b)
        out_parents[b] = m->body_parentid[b];
    return AISTANK_OK;
}

AISTANK_API AistankResult Engine_GetAgentBodyPositions(const AistankEngine* e,
                                                       uint32_t agent, float* out_xyz) {
    if (!e || !out_xyz) return AISTANK_ERR_BAD_ARG;
    auto& envs = const_cast<AistankEngine*>(e)->core->Envs();
    if (agent >= envs.size()) return AISTANK_ERR_BAD_ARG;
    const mjModel* m = e->core->Model();
    const mjData* d = envs[agent].data;
    for (int b = 0; b < m->nbody; ++b) {
        out_xyz[b * 3 + 0] = static_cast<float>(d->xpos[b * 3 + 0]);
        out_xyz[b * 3 + 1] = static_cast<float>(d->xpos[b * 3 + 1]);
        out_xyz[b * 3 + 2] = static_cast<float>(d->xpos[b * 3 + 2]);
    }
    return AISTANK_OK;
}

AISTANK_API uint32_t Engine_GetGeomCount(const AistankEngine* e) {
    return e ? static_cast<uint32_t>(e->core->Model()->ngeom) : 0;
}

// Static per-geom description (constant for the run): MuJoCo geom type enum and
// size[3]. type: 0=plane 1=hfield 2=sphere 3=capsule 4=ellipsoid 5=cylinder 6=box.
AISTANK_API AistankResult Engine_GetGeomStatic(const AistankEngine* e,
                                               int32_t* out_types, float* out_sizes) {
    if (!e || !out_types || !out_sizes) return AISTANK_ERR_BAD_ARG;
    const mjModel* m = e->core->Model();
    for (int g = 0; g < m->ngeom; ++g) {
        out_types[g] = m->geom_type[g];
        out_sizes[g * 3 + 0] = static_cast<float>(m->geom_size[g * 3 + 0]);
        out_sizes[g * 3 + 1] = static_cast<float>(m->geom_size[g * 3 + 1]);
        out_sizes[g * 3 + 2] = static_cast<float>(m->geom_size[g * 3 + 2]);
    }
    return AISTANK_OK;
}

// One agent's per-geom world transform this frame: position xyz (ngeom*3) and
// row-major 3x3 orientation matrix (ngeom*9), straight from mjData.
AISTANK_API AistankResult Engine_GetAgentGeomPose(const AistankEngine* e, uint32_t agent,
                                                  float* out_xpos, float* out_xmat) {
    if (!e || !out_xpos || !out_xmat) return AISTANK_ERR_BAD_ARG;
    auto& envs = const_cast<AistankEngine*>(e)->core->Envs();
    if (agent >= envs.size()) return AISTANK_ERR_BAD_ARG;
    const mjModel* m = e->core->Model();
    const mjData* d = envs[agent].data;
    for (int g = 0; g < m->ngeom; ++g) {
        for (int i = 0; i < 3; ++i) out_xpos[g * 3 + i] = static_cast<float>(d->geom_xpos[g * 3 + i]);
        for (int i = 0; i < 9; ++i) out_xmat[g * 9 + i] = static_cast<float>(d->geom_xmat[g * 9 + i]);
    }
    return AISTANK_OK;
}

AISTANK_API uint32_t Engine_GetObservationDim(const AistankEngine* e) {
    return e ? e->core->ObsDim() : 0;
}

AISTANK_API uint32_t Engine_GetActionDim(const AistankEngine* e) {
    return e ? e->core->ActDim() : 0;
}

AISTANK_API const char* Engine_GetLastError(const AistankEngine* e) {
    return e ? e->lastError.c_str() : "";
}

} // extern "C"
