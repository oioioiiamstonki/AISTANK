// GpuBuffers.h — SoA simulation state buffers + upload/readback rings.
// All per-agent state is Structure-of-Arrays: flat float buffers indexed
// [component * numAgents + agent] so 64-lane threadgroups issue coalesced loads.
#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <stdexcept>

namespace aistank {

using Microsoft::WRL::ComPtr;

constexpr uint32_t kFramesInFlight = 2;

// Fixed descriptor-table slots (must match RootSig.hlsli register mapping).
enum SrvSlot : uint32_t {
    SRV_JointPos = 0, SRV_JointVel, SRV_RootPos, SRV_RootQuat,
    SRV_RootLinVel, SRV_RootAngVel, SRV_InitialState, SRV_Count
};
enum UavSlot : uint32_t {
    UAV_Observations = 0, UAV_Actions, UAV_Rewards, UAV_DoneFlags,
    UAV_StepCount, UAV_RngState, UAV_RolloutObs, UAV_RolloutAct,
    UAV_RolloutRew, UAV_RolloutDone, UAV_RolloutVal, UAV_EpisodeStats,
    UAV_InstanceXforms, UAV_Count
};
enum WeightSlot : uint32_t { SRV_Weights0 = 32, SRV_Weights1 = 33 };

struct BufferDesc { uint64_t elements; uint32_t stride; };

// One default-heap structured buffer with creation-time SRV/UAV registration.
struct GpuBuffer {
    ComPtr<ID3D12Resource> res;
    uint64_t byteSize = 0;

    void Create(ID3D12Device* dev, uint64_t elements, uint32_t stride,
                D3D12_RESOURCE_STATES initial, bool uav) {
        byteSize = elements * stride;
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = byteSize; rd.Height = 1; rd.DepthOrArraySize = 1;
        rd.MipLevels = 1; rd.SampleDesc = {1, 0};
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                       : D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                initial, nullptr, IID_PPV_ARGS(&res))))
            throw std::runtime_error("GpuBuffer::Create failed");
    }
};

// Persistently-mapped upload (or readback) ring, one slot per frame in flight.
// A slot is reused only after its fence value has been reached — enforced by
// SimulationLoop's frame throttle, so no per-buffer bookkeeping is needed here.
struct StagingRing {
    ComPtr<ID3D12Resource> res[kFramesInFlight];
    uint8_t* mapped[kFramesInFlight]{};
    uint64_t slotBytes = 0;

    void Create(ID3D12Device* dev, uint64_t bytes, D3D12_HEAP_TYPE type) {
        slotBytes = bytes;
        D3D12_HEAP_PROPERTIES hp{ type };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1;
        rd.MipLevels = 1; rd.SampleDesc = {1, 0};
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_RESOURCE_STATES state = (type == D3D12_HEAP_TYPE_UPLOAD)
            ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                    state, nullptr, IID_PPV_ARGS(&res[i]))))
                throw std::runtime_error("StagingRing::Create failed");
            D3D12_RANGE noRead{0, 0};
            res[i]->Map(0, &noRead, reinterpret_cast<void**>(&mapped[i]));
        }
    }
};

// All simulation-side GPU memory for N agents / D dof / A actions / O obs dims.
struct SimBuffers {
    uint32_t N = 0, D = 0, A = 0, O = 0, horizon = 0;

    // Physics mirror (copy-dest → SRV for compute)
    GpuBuffer jointPos, jointVel;          // float[D * N] each
    GpuBuffer rootPos, rootLinVel, rootAngVel; // float[3 * N]
    GpuBuffer rootQuat;                    // float[4 * N]
    GpuBuffer initialState;                // packed reset pose, float[(2D + 13) * N]

    // RL state (UAVs)
    GpuBuffer observations;                // float[O * N]
    GpuBuffer actions;                     // float[A * N]
    GpuBuffer rewards;                     // float[N]
    GpuBuffer doneFlags, stepCount, rngState; // uint[N]
    GpuBuffer values;                      // float[N]
    GpuBuffer episodeStats;                // float[4] (atomics: Σreward, Σlen, #resets, pad)

    // Rollout storage, appended each tick at offset (tick % horizon)
    GpuBuffer rolloutObs, rolloutAct, rolloutRew, rolloutDone, rolloutVal;

    // Policy weights, double-buffered (slot selected via root constant)
    GpuBuffer weights[2];                  // float[paramCount]

    // Preview
    GpuBuffer instanceXforms;              // float4x4 per previewed body

    // PCIe boundary
    StagingRing stateUpload;               // CPU → GPU physics mirror, SoA-packed
    StagingRing actionReadback;            // GPU → CPU actions for mj applyCtrl
    StagingRing rolloutReadback;           // GPU → CPU once per horizon
    StagingRing weightUpload;

    void Create(ID3D12Device* dev, uint32_t numAgents, uint32_t dof,
                uint32_t actDim, uint32_t obsDim, uint32_t rolloutHorizon,
                uint64_t policyParamCount);
};

} // namespace aistank
