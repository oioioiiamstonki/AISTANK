// EngineCore.cpp — DX12 + MuJoCo initialization with explicit lifetime handling.
#include "EngineCore.h"

#include <cassert>
#include <fstream>
#include <stdexcept>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace aistank {

static void ThrowIfFailed(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char buf[256];
        snprintf(buf, sizeof buf, "%s (hr=0x%08lX)", what, static_cast<unsigned long>(hr));
        throw std::runtime_error(buf);
    }
}

// ---------------------------------------------------------------- FencedQueue

void FencedQueue::Create(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type) {
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = type;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    ThrowIfFailed(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ThrowIfFailed(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
}

uint64_t FencedQueue::Signal() {
    ++lastSignaled;
    ThrowIfFailed(queue->Signal(fence.Get(), lastSignaled), "Queue::Signal");
    return lastSignaled;
}

void FencedQueue::GpuWait(const FencedQueue& other, uint64_t value) {
    ThrowIfFailed(queue->Wait(other.fence.Get(), value), "Queue::Wait");
}

void FencedQueue::CpuWait(uint64_t value) {
    if (fence->GetCompletedValue() < value) {
        ThrowIfFailed(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion");
        WaitForSingleObject(event, INFINITE);
    }
}

void FencedQueue::Destroy() {
    if (event) { CloseHandle(event); event = nullptr; }
}

// ----------------------------------------------------------------- SimBuffers

void SimBuffers::Create(ID3D12Device* dev, uint32_t numAgents, uint32_t dof,
                        uint32_t actDim, uint32_t obsDim, uint32_t rolloutHorizon,
                        uint64_t policyParamCount) {
    N = numAgents; D = dof; A = actDim; O = obsDim; horizon = rolloutHorizon;
    constexpr auto kSrvState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    constexpr auto kUavState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    // Physics mirror — copy targets, read as SRVs by compute.
    jointPos   .Create(dev, uint64_t(D) * N, 4, kSrvState, false);
    jointVel   .Create(dev, uint64_t(D) * N, 4, kSrvState, false);
    rootPos    .Create(dev, 3ull * N, 4, kSrvState, false);
    rootQuat   .Create(dev, 4ull * N, 4, kSrvState, false);
    rootLinVel .Create(dev, 3ull * N, 4, kSrvState, false);
    rootAngVel .Create(dev, 3ull * N, 4, kSrvState, false);
    initialState.Create(dev, (2ull * D + 13) * N, 4, kSrvState, false);

    // RL working set — UAVs.
    observations.Create(dev, uint64_t(O) * N, 4, kUavState, true);
    actions     .Create(dev, uint64_t(A) * N, 4, kUavState, true);
    rewards     .Create(dev, N, 4, kUavState, true);
    values      .Create(dev, N, 4, kUavState, true);
    doneFlags   .Create(dev, N, 4, kUavState, true);
    stepCount   .Create(dev, N, 4, kUavState, true);
    rngState    .Create(dev, N, 4, kUavState, true);
    episodeStats.Create(dev, 4, 4, kUavState, true);

    // Rollout storage: [horizon][component][agent].
    rolloutObs .Create(dev, uint64_t(horizon) * O * N, 4, kUavState, true);
    rolloutAct .Create(dev, uint64_t(horizon) * A * N, 4, kUavState, true);
    rolloutRew .Create(dev, uint64_t(horizon) * N, 4, kUavState, true);
    rolloutDone.Create(dev, uint64_t(horizon) * N, 4, kUavState, true);
    rolloutVal .Create(dev, uint64_t(horizon) * N, 4, kUavState, true);

    weights[0].Create(dev, policyParamCount, 4, kSrvState, false);
    weights[1].Create(dev, policyParamCount, 4, kSrvState, false);

    instanceXforms.Create(dev, 64ull * 16 /*float4x4, first 64 agents*/, 4, kUavState, true);

    // PCIe staging. Upload slot holds one full SoA physics snapshot.
    const uint64_t stateBytes   = (uint64_t(2 * D) + 13) * N * 4;
    const uint64_t actionBytes  = uint64_t(A) * N * 4;
    const uint64_t rolloutBytes = uint64_t(horizon) * (O + A + 3) * N * 4;
    stateUpload    .Create(dev, stateBytes,             D3D12_HEAP_TYPE_UPLOAD);
    actionReadback .Create(dev, actionBytes,            D3D12_HEAP_TYPE_READBACK);
    rolloutReadback.Create(dev, rolloutBytes,           D3D12_HEAP_TYPE_READBACK);
    weightUpload   .Create(dev, policyParamCount * 4,   D3D12_HEAP_TYPE_UPLOAD);
}

// ------------------------------------------------------------------ EngineCore

EngineCore::EngineCore(const AistankConfig& cfg, const char* mjcfPath) : m_cfg(cfg) {
    if (cfg.num_agents == 0) throw std::runtime_error("num_agents must be > 0");
    CreateDevice();
    CreateQueues();
    LoadMuJoCo(mjcfPath);            // sets m_dof / m_actDim / m_obsDim first —
    CreateRootSignatures();          // buffer sizes depend on the model
    LoadPipelines();
    CreateBuffersAndDescriptors();
    UploadInitialState();
}

EngineCore::~EngineCore() {
    // Drain all queues before releasing resources the GPU may still reference.
    for (FencedQueue* q : { &m_computeQueue, &m_copyQueue, &m_directQueue }) {
        if (q->queue) q->CpuWait(q->Signal());
        q->Destroy();
    }
    for (EnvSlot& env : m_envs)
        if (env.data) mj_deleteData(env.data);
    if (m_model) mj_deleteModel(m_model);
    // ComPtr handles the rest in member-reverse order.
}

void EngineCore::CreateDevice() {
#if defined(_DEBUG)
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        debug->EnableDebugLayer();
#endif
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory)), "CreateDXGIFactory2");

    // Prefer the highest-performance adapter (discrete GPU).
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;
         m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                               IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i) {
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                        IID_PPV_ARGS(&m_device))))
            break;
        adapter.Reset();
    }
    if (!m_device) throw std::runtime_error("No DX12 FL12.0 adapter found");
}

void EngineCore::CreateQueues() {
    m_computeQueue.Create(m_device.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE);
    m_copyQueue   .Create(m_device.Get(), D3D12_COMMAND_LIST_TYPE_COPY);
    m_directQueue .Create(m_device.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
}

void EngineCore::CreateRootSignatures() {
    // [0] root CBV b0 (SimConstants)
    // [1] SRV table t0..t15 + t32..t33 (weights)
    // [2] UAV table u0..u15
    // [3] 4 root constants b1 (passIndex, layerIndex, weightSlot, rngSeed)
    D3D12_DESCRIPTOR_RANGE srvRanges[2] = {};
    srvRanges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 16, 0, 0,
                     D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND };
    srvRanges[1] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 32, 0,
                     D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND };
    D3D12_DESCRIPTOR_RANGE uavRange =
        { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 16, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND };

    D3D12_ROOT_PARAMETER params[4] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor = { /*b*/0, /*space*/0 };
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = { 2, srvRanges };
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = { 1, &uavRange };
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants = { /*b*/1, /*space*/0, /*num32bit*/4 };
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 4;
    rsd.pParameters = params;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                              &blob, &err), "SerializeRootSignature");
    ThrowIfFailed(m_device->CreateRootSignature(0, blob->GetBufferPointer(),
                                                blob->GetBufferSize(),
                                                IID_PPV_ARGS(&m_rootSigCompute)),
                  "CreateRootSignature(compute)");
    // Graphics root signature for the preview renderer is created lazily in the
    // renderer module when enable_preview is set (omitted from the scaffold core).
}

static std::vector<uint8_t> LoadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader blob not found: " + path);
    std::vector<uint8_t> data(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

void EngineCore::LoadPipelines() {
    auto makeCs = [&](const char* cso, ComPtr<ID3D12PipelineState>& out) {
        std::vector<uint8_t> blob = LoadFile(std::string("shaders/") + cso);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = m_rootSigCompute.Get();
        pd.CS = { blob.data(), blob.size() };
        ThrowIfFailed(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&out)),
                      "CreateComputePipelineState");
    };
    makeCs("CS_GatherObservations.cso", m_pipelines.observe);
    makeCs("CS_PolicyForward.cso",      m_pipelines.policyForward);
    makeCs("CS_RewardAndTerminate.cso", m_pipelines.rewardTerminate);
    makeCs("CS_ApplyResets.cso",        m_pipelines.applyResets);
    makeCs("CS_BuildInstances.cso",     m_pipelines.buildInstances);
}

void EngineCore::LoadMuJoCo(const char* mjcfPath) {
    char error[512] = {};
    m_model = mj_loadXML(mjcfPath, nullptr, error, sizeof error);
    if (!m_model) throw std::runtime_error(std::string("mj_loadXML: ") + error);

    m_dof    = static_cast<uint32_t>(m_model->nv) - 6;   // exclude free-joint root
    m_actDim = static_cast<uint32_t>(m_model->nu);
    // obs: joint pos + joint vel + root height + root quat + root lin/ang vel + prev actions
    m_obsDim = 2 * m_dof + 1 + 4 + 6 + m_actDim;
    m_policyParamCount =
        uint64_t(m_obsDim) * 256 + 256 +      // layer 1
        256ull * 256 + 256 +                  // layer 2
        256ull * (m_actDim + 1) + m_actDim + 1; // heads: mu + value

    m_envs.resize(m_cfg.num_agents);
    for (EnvSlot& env : m_envs) {
        env.data = mj_makeData(m_model);
        if (!env.data) throw std::runtime_error("mj_makeData failed");
        mj_resetData(m_model, env.data);
        mj_forward(m_model, env.data);  // populate derived quantities for tick 0
    }
}

void EngineCore::CreateBuffersAndDescriptors() {
    m_buffers.Create(m_device.Get(), m_cfg.num_agents, m_dof, m_actDim, m_obsDim,
                     m_cfg.rollout_horizon ? m_cfg.rollout_horizon : 32,
                     m_policyParamCount);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 4096;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_srvUavHeap)),
                  "CreateDescriptorHeap");

    const UINT inc =
        m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    auto srvAt = [&](uint32_t slot, GpuBuffer& b, uint64_t elements) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer = { 0, static_cast<UINT>(elements), 4, D3D12_BUFFER_SRV_FLAG_NONE };
        m_device->CreateShaderResourceView(b.res.Get(), &d,
            { cpu.ptr + uint64_t(slot) * inc });
    };
    auto uavAt = [&](uint32_t slot, GpuBuffer& b, uint64_t elements) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer = { 0, static_cast<UINT>(elements), 4, 0, D3D12_BUFFER_UAV_FLAG_NONE };
        m_device->CreateUnorderedAccessView(b.res.Get(), nullptr, &d,
            { cpu.ptr + uint64_t(16 + slot) * inc });  // UAV table starts at heap slot 16
    };

    const SimBuffers& B = m_buffers;
    const uint64_t N = B.N;
    srvAt(SRV_JointPos,     m_buffers.jointPos,     uint64_t(B.D) * N);
    srvAt(SRV_JointVel,     m_buffers.jointVel,     uint64_t(B.D) * N);
    srvAt(SRV_RootPos,      m_buffers.rootPos,      3 * N);
    srvAt(SRV_RootQuat,     m_buffers.rootQuat,     4 * N);
    srvAt(SRV_RootLinVel,   m_buffers.rootLinVel,   3 * N);
    srvAt(SRV_RootAngVel,   m_buffers.rootAngVel,   3 * N);
    srvAt(SRV_InitialState, m_buffers.initialState, (2ull * B.D + 13) * N);

    uavAt(UAV_Observations, m_buffers.observations, uint64_t(B.O) * N);
    uavAt(UAV_Actions,      m_buffers.actions,      uint64_t(B.A) * N);
    uavAt(UAV_Rewards,      m_buffers.rewards,      N);
    uavAt(UAV_DoneFlags,    m_buffers.doneFlags,    N);
    uavAt(UAV_StepCount,    m_buffers.stepCount,    N);
    uavAt(UAV_RngState,     m_buffers.rngState,     N);
    uavAt(UAV_RolloutObs,   m_buffers.rolloutObs,   uint64_t(B.horizon) * B.O * N);
    uavAt(UAV_RolloutAct,   m_buffers.rolloutAct,   uint64_t(B.horizon) * B.A * N);
    uavAt(UAV_RolloutRew,   m_buffers.rolloutRew,   uint64_t(B.horizon) * N);
    uavAt(UAV_RolloutDone,  m_buffers.rolloutDone,  uint64_t(B.horizon) * N);
    uavAt(UAV_RolloutVal,   m_buffers.rolloutVal,   uint64_t(B.horizon) * N);
    uavAt(UAV_EpisodeStats, m_buffers.episodeStats, 4);
    uavAt(UAV_InstanceXforms, m_buffers.instanceXforms, 64ull * 16);

    // Weight slots live at heap slots 32/33 (second SRV range, registers t32/t33).
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer = { 0, static_cast<UINT>(m_policyParamCount), 4, D3D12_BUFFER_SRV_FLAG_NONE };
        for (int s = 0; s < 2; ++s)
            m_device->CreateShaderResourceView(m_buffers.weights[s].res.Get(), &d,
                { cpu.ptr + uint64_t(32 + s) * inc });
    }

    m_srvTableGpu = { gpu.ptr };                       // table starts at slot 0
    m_uavTableGpu = { gpu.ptr + 16ull * inc };         // UAV range starts at slot 16
}

void EngineCore::UploadInitialState() {
    // Pack each env's reset pose (qpos/qvel + root state) into initialState, SoA.
    // Recorded on the copy queue and fenced before the first Tick. Implementation
    // mirrors SimulationLoop::PackPhysicsState — see SimulationLoop.cpp.
}

void EngineCore::CheckDeviceRemoved() const {
    HRESULT hr = m_device->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        char buf[128];
        snprintf(buf, sizeof buf, "DX12 device removed (0x%08lX)",
                 static_cast<unsigned long>(hr));
        throw std::runtime_error(buf);
    }
}

} // namespace aistank
