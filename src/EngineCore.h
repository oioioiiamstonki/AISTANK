// EngineCore.h — DX12 device/queues/heaps/PSOs + MuJoCo global context.
#pragma once
#include "GpuBuffers.h"
#include "../include/aistank.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <string>
#include <vector>

namespace aistank {

using Microsoft::WRL::ComPtr;

// Per-queue fence wrapper. Signal() returns the value to wait on.
struct FencedQueue {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence>        fence;
    uint64_t                   lastSignaled = 0;
    HANDLE                     event = nullptr;

    void Create(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type);
    uint64_t Signal();
    void GpuWait(const FencedQueue& other, uint64_t value); // cross-queue dependency
    void CpuWait(uint64_t value);                           // throttle only
    void Destroy();
};

// Compiled compute pipelines, all sharing m_rootSigCompute.
struct ComputePipelines {
    ComPtr<ID3D12PipelineState> observe;
    ComPtr<ID3D12PipelineState> policyForward;
    ComPtr<ID3D12PipelineState> rewardTerminate;
    ComPtr<ID3D12PipelineState> applyResets;
    ComPtr<ID3D12PipelineState> buildInstances;  // sim state → preview transforms
};

// Per-env CPU physics slot. mjModel is shared (read-only after load);
// mjData is per-env, making mj_step embarrassingly parallel.
struct EnvSlot {
    mjData* data = nullptr;
    uint32_t stepsSinceReset = 0;
};

class EngineCore {
public:
    EngineCore(const AistankConfig& cfg, const char* mjcfPath); // throws std::runtime_error
    ~EngineCore();
    EngineCore(const EngineCore&) = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    // --- accessors used by SimulationLoop ---
    ID3D12Device*           Device()        { return m_device.Get(); }
    FencedQueue&            ComputeQueue()  { return m_computeQueue; }
    FencedQueue&            CopyQueue()     { return m_copyQueue; }
    FencedQueue&            DirectQueue()   { return m_directQueue; }
    ComputePipelines&       Pipelines()     { return m_pipelines; }
    ID3D12RootSignature*    ComputeRootSig(){ return m_rootSigCompute.Get(); }
    ID3D12DescriptorHeap*   SrvUavHeap()    { return m_srvUavHeap.Get(); }
    SimBuffers&             Buffers()       { return m_buffers; }
    const mjModel*          Model() const   { return m_model; }
    std::vector<EnvSlot>&   Envs()          { return m_envs; }
    const AistankConfig&    Config() const  { return m_cfg; }
    uint32_t ObsDim() const { return m_obsDim; }
    uint32_t ActDim() const { return m_actDim; }
    uint64_t PolicyParamCount() const { return m_policyParamCount; }

    D3D12_GPU_DESCRIPTOR_HANDLE SrvTableGpu() const { return m_srvTableGpu; }
    D3D12_GPU_DESCRIPTOR_HANDLE UavTableGpu() const { return m_uavTableGpu; }

    void CheckDeviceRemoved() const; // throws with DRED-ish diagnostics

private:
    void CreateDevice();
    void CreateQueues();
    void CreateRootSignatures();
    void LoadPipelines();           // loads shaders/*.cso built by CMake
    void LoadMuJoCo(const char* mjcfPath);
    void CreateBuffersAndDescriptors();
    void UploadInitialState();

    AistankConfig m_cfg;
    uint32_t m_obsDim = 0, m_actDim = 0, m_dof = 0;
    uint64_t m_policyParamCount = 0;

    // --- DX12 ---
    ComPtr<IDXGIFactory6>        m_factory;
    ComPtr<ID3D12Device>         m_device;
    FencedQueue                  m_computeQueue, m_copyQueue, m_directQueue;
    ComPtr<ID3D12RootSignature>  m_rootSigCompute;
    ComPtr<ID3D12RootSignature>  m_rootSigGraphics;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;     // single shader-visible heap
    D3D12_GPU_DESCRIPTOR_HANDLE  m_srvTableGpu{}, m_uavTableGpu{};
    ComputePipelines             m_pipelines;
    ComPtr<ID3D12PipelineState>  m_psoPreviewGfx;
    SimBuffers                   m_buffers;

    // --- MuJoCo ---
    mjModel*             m_model = nullptr;       // shared, read-only after load
    std::vector<EnvSlot> m_envs;                  // one mjData per agent

    std::string m_lastError;
    friend struct ::AistankEngine;
};

} // namespace aistank
