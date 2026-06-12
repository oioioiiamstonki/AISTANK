// SimulationLoop.h — the per-tick orchestration: physics → upload → compute chain → readback.
#pragma once
#include "EngineCore.h"
#include <atomic>
#include <thread>
#include <vector>

namespace aistank {

// Seam for swapping the hand-rolled HLSL MLP for DirectML/onnxruntime-dml later.
class IPolicyExecutor {
public:
    virtual ~IPolicyExecutor() = default;
    // Records inference dispatches into an already-open compute command list.
    // Must read UAV_Observations and write UAV_Actions / values.
    virtual void RecordInference(ID3D12GraphicsCommandList* cl,
                                 uint32_t weightSlot) = 0;
};

// Default executor: CS_PolicyForward, one threadgroup per agent.
class HlslMlpExecutor final : public IPolicyExecutor {
public:
    explicit HlslMlpExecutor(EngineCore& core) : m_core(core) {}
    void RecordInference(ID3D12GraphicsCommandList* cl, uint32_t weightSlot) override;
private:
    EngineCore& m_core;
};

// Matches SimConstants cbuffer in shaders/AgentShaders.hlsl (16-byte aligned).
struct alignas(256) SimConstants {
    uint32_t numAgents;
    uint32_t dofCount;
    uint32_t actionDim;
    uint32_t obsDim;
    float    controlDt;
    float    targetVelocity;
    float    terminationHeight;
    float    wForward;
    float    wAlive;
    float    wEnergy;
    float    wUpright;
    uint32_t maxEpisodeSteps;
    uint32_t rolloutHorizon;
    uint32_t rolloutCursor;     // tick % horizon, updated per frame via root constant instead
    uint32_t pad[2];
};

class SimulationLoop {
public:
    explicit SimulationLoop(EngineCore& core);
    ~SimulationLoop();

    // One control step. See .cpp for the exact sequence.
    AistankResult Tick(AistankStepStats& outStats);

    // Trainer interface (called by the C ABI layer).
    AistankResult SetPolicyWeights(const float* params, uint64_t count);
    AistankResult MapRollout(const float** obs, const float** act, const float** rew,
                             const uint8_t** done, const float** val,
                             uint32_t* horizon, uint32_t* obsDim, uint32_t* actDim);

private:
    // Phase A (CPU, worker pool): apply previous actions, mj_step × substeps,
    // scatter qpos/qvel into the SoA upload slot.
    void StepPhysicsAndPack(uint32_t frameSlot, float* outPhysicsMs);
    // Phase B (copy queue): staged SoA snapshot → default-heap physics mirror.
    void RecordUpload(uint32_t frameSlot);
    // Phase C (compute queue): observe → policy → reward/terminate → resets → instances.
    void RecordComputeChain(uint32_t frameSlot);
    // Phase D (copy queue): actions → readback ring; rollout → readback at horizon end.
    void RecordReadback(uint32_t frameSlot, bool horizonEnd);
    // CPU-side env reset for agents the GPU flagged done last tick.
    void ApplyCpuResets(uint32_t frameSlot);

    EngineCore& m_core;
    std::unique_ptr<IPolicyExecutor> m_policy;

    // Per-frame-in-flight command allocators/lists (compute + copy).
    ComPtr<ID3D12CommandAllocator>    m_computeAlloc[kFramesInFlight];
    ComPtr<ID3D12GraphicsCommandList> m_computeList[kFramesInFlight];
    ComPtr<ID3D12CommandAllocator>    m_copyAlloc[kFramesInFlight];
    ComPtr<ID3D12GraphicsCommandList> m_copyList[kFramesInFlight];

    // Persistent CBV (one slot per frame in flight, persistently mapped).
    ComPtr<ID3D12Resource> m_constantsBuffer;
    SimConstants*          m_constantsMapped = nullptr;

    // GPU timestamp queries for gpu_compute_ms.
    ComPtr<ID3D12QueryHeap> m_tsHeap;
    ComPtr<ID3D12Resource>  m_tsReadback;
    uint64_t                m_tsFrequency = 0;

    // Fence values guarding each ring slot.
    uint64_t m_copyFenceAt[kFramesInFlight]{};
    uint64_t m_computeFenceAt[kFramesInFlight]{};

    // CPU mirror of done flags from the *previous* tick's readback (drives mj_resetData).
    std::vector<uint8_t> m_pendingResets;

    uint64_t m_tick = 0;
    uint32_t m_activeWeightSlot = 0;
    std::atomic<uint32_t> m_pendingWeightSlot{ UINT32_MAX };

    // Worker pool for mj_step.
    std::vector<std::thread> m_workers;
    // (work distribution: simple atomic counter over env indices; see .cpp)
};

} // namespace aistank
