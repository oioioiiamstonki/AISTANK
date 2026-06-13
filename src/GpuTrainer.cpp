// GpuTrainer.cpp — see GpuTrainer.h. PPO update as a chain of compute dispatches.
#include "GpuTrainer.h"
#include "EngineCore.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace aistank {

static constexpr uint32_t HD = 256;
static constexpr uint32_t kMaxGroups = 65535;

static void TIF(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(what);
}

static std::string ModuleDirT() {
    HMODULE mod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&ModuleDirT), &mod);
    char path[MAX_PATH] = {};
    GetModuleFileNameA(mod, path, MAX_PATH);
    std::string s(path);
    size_t slash = s.find_last_of("\\/");
    return slash == std::string::npos ? std::string(".") : s.substr(0, slash);
}

static std::vector<uint8_t> LoadBlob(const std::string& name) {
    std::string path = ModuleDirT() + "/shaders/" + name;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("train shader not found: " + path);
    std::vector<uint8_t> d(static_cast<size_t>(f.tellg()));
    f.seekg(0); f.read(reinterpret_cast<char*>(d.data()), d.size());
    return d;
}

GpuTrainer::GpuTrainer(EngineCore& core) : m_core(core), m_dev(core.Device()) {
    SimBuffers& B = core.Buffers();
    m_O = core.ObsDim(); m_A = core.ActDim(); m_Aout = m_A + 1;
    m_N = B.N; m_H = B.horizon; m_P = core.PolicyParamCount();
    m_S = (m_H - 1) * m_N;

    CreateBuffers();
    CreateRootSigAndPipelines();

    TIF(m_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&m_alloc)), "trainer alloc");
    TIF(m_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_alloc.Get(),
        nullptr, IID_PPV_ARGS(&m_list)), "trainer list");
    m_list->Close();

    WriteDescriptors(B.weights[0].res.Get());

    // Constants (set once).
    {
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = sizeof(TrainConstantsCB);
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc = {1, 0}; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        TIF(m_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_constants)),
            "train constants");
        D3D12_RANGE nr{0, 0};
        m_constants->Map(0, &nr, reinterpret_cast<void**>(&m_constMapped));
    }
    const uint32_t w1 = HD * m_O, w2 = HD * HD, w3 = m_Aout * HD;
    TrainConstantsCB& c = *m_constMapped;
    c.S = m_S; c.O = m_O; c.A = m_A; c.Aout = m_Aout; c.N = m_N; c.H = m_H;
    c.P = static_cast<uint32_t>(m_P);
    c.OffW1 = 0; c.OffB1 = w1; c.OffW2 = w1 + HD; c.OffB2 = w1 + HD + w2;
    c.OffW3 = w1 + HD + w2 + HD; c.OffB3 = w1 + HD + w2 + HD + w3;
    c.Gamma = 0.99f; c.Lambda = 0.95f; c.ClipEps = 0.2f; c.ValueCoef = 0.5f;
    c.Sigma = 0.2f; c.Lr = 3e-4f; c.Beta1 = 0.9f; c.Beta2 = 0.999f; c.AdamEps = 1e-8f;

    // Readback buffers.
    auto makeReadback = [&](uint64_t bytes, ComPtr<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc = {1, 0}; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        TIF(m_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out)), "readback");
    };
    makeReadback(8 * sizeof(uint32_t), m_statsReadback);
    makeReadback(m_P * 4, m_weightsReadback);
    makeReadback(m_P * 4, m_gradReadback);
}

void GpuTrainer::CreateBuffers() {
    auto mk = [&](GpuBuffer& b, uint64_t elems) {
        b.Create(m_dev, elems, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    };
    mk(m_grad, m_P); mk(m_adamM, m_P); mk(m_adamV, m_P);
    mk(m_adv, m_S); mk(m_vtarget, m_S); mk(m_oldLogp, m_S);
    mk(m_xbuf, uint64_t(m_S) * m_O);
    mk(m_h1, uint64_t(m_S) * HD); mk(m_h2, uint64_t(m_S) * HD);
    mk(m_dz1, uint64_t(m_S) * HD); mk(m_dz2, uint64_t(m_S) * HD);
    mk(m_dz3, uint64_t(m_S) * m_Aout);
    mk(m_stats, 8);
}

void GpuTrainer::CreateRootSigAndPipelines() {
    D3D12_DESCRIPTOR_RANGE uav{ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 19, 0, 0,
                                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND };
    D3D12_ROOT_PARAMETER p[3] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    p[0].Descriptor = { 0, 0 };
    p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[1].Constants = { 1, 0, 4 };
    p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p[2].DescriptorTable = { 1, &uav };
    p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 3; rsd.pParameters = p;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> blob, err;
    TIF(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err),
        "train rootsig serialize");
    TIF(m_dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)), "train rootsig");

    auto cs = [&](const char* name, ComPtr<ID3D12PipelineState>& out) {
        std::vector<uint8_t> b = LoadBlob(name);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = m_rootSig.Get();
        pd.CS = { b.data(), b.size() };
        TIF(m_dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&out)), "train pso");
    };
    cs("CS_GAE.cso", m_pso.gae);
    cs("CS_AdvNorm.cso", m_pso.advNorm);
    cs("CS_TrainForward.cso", m_pso.forward);
    cs("CS_TrainBackward.cso", m_pso.backward);
    cs("CS_WeightGrad.cso", m_pso.weightGrad);
    cs("CS_BiasGrad.cso", m_pso.biasGrad);
    cs("CS_Adam.cso", m_pso.adam);
    cs("CS_ZeroStats.cso", m_pso.zeroStats);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 19;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    TIF(m_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)), "train heap");
    m_inc = m_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void GpuTrainer::WriteDescriptors(ID3D12Resource* weights) {
    SimBuffers& B = m_core.Buffers();
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_heap->GetCPUDescriptorHandleForHeapStart();
    uint32_t slot = 0;
    auto uav = [&](ID3D12Resource* res, uint64_t elems) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer = { 0, static_cast<UINT>(elems), 4, 0, D3D12_BUFFER_UAV_FLAG_NONE };
        m_dev->CreateUnorderedAccessView(res, nullptr, &d, { cpu.ptr + uint64_t(slot++) * m_inc });
    };
    uav(weights, m_P);                                  // u0
    uav(m_grad.res.Get(), m_P);                         // u1
    uav(m_adamM.res.Get(), m_P);                        // u2
    uav(m_adamV.res.Get(), m_P);                        // u3
    uav(B.rolloutObs.res.Get(),  uint64_t(m_H) * m_O * m_N); // u4
    uav(B.rolloutAct.res.Get(),  uint64_t(m_H) * m_A * m_N); // u5
    uav(B.rolloutRew.res.Get(),  uint64_t(m_H) * m_N);       // u6
    uav(B.rolloutDone.res.Get(), uint64_t(m_H) * m_N);       // u7
    uav(B.rolloutVal.res.Get(),  uint64_t(m_H) * m_N);       // u8
    uav(m_adv.res.Get(), m_S);                          // u9
    uav(m_vtarget.res.Get(), m_S);                      // u10
    uav(m_oldLogp.res.Get(), m_S);                      // u11
    uav(m_xbuf.res.Get(), uint64_t(m_S) * m_O);         // u12
    uav(m_h1.res.Get(), uint64_t(m_S) * HD);            // u13
    uav(m_h2.res.Get(), uint64_t(m_S) * HD);            // u14
    uav(m_dz1.res.Get(), uint64_t(m_S) * HD);           // u15
    uav(m_dz2.res.Get(), uint64_t(m_S) * HD);           // u16
    uav(m_dz3.res.Get(), uint64_t(m_S) * m_Aout);       // u17
    uav(m_stats.res.Get(), 8);                          // u18
    m_boundWeights = weights;
}

void GpuTrainer::Begin() {
    m_alloc->Reset();
    m_list->Reset(m_alloc.Get(), nullptr);
    ID3D12DescriptorHeap* heaps[] = { m_heap.Get() };
    m_list->SetDescriptorHeaps(1, heaps);
    m_list->SetComputeRootSignature(m_rootSig.Get());
    m_list->SetComputeRootConstantBufferView(0, m_constants->GetGPUVirtualAddress());
    m_list->SetComputeRootDescriptorTable(2, m_heap->GetGPUDescriptorHandleForHeapStart());
}

void GpuTrainer::SetPass(uint32_t layer, uint32_t adamT, uint32_t flags) {
    // SBase is set separately by Dispatch loops via this same root slot.
    uint32_t c[4] = { layer, adamT, flags, 0 };
    m_list->SetComputeRoot32BitConstants(1, 4, c, 0);
}

void GpuTrainer::Dispatch(ID3D12PipelineState* pso, uint32_t gx, uint32_t gy) {
    m_list->SetPipelineState(pso);
    m_list->Dispatch(gx, gy, 1);
}

void GpuTrainer::UavBarrier() {
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    m_list->ResourceBarrier(1, &b);
}

void GpuTrainer::ExecuteAndWait() {
    m_list->Close();
    ID3D12CommandList* lists[] = { m_list.Get() };
    m_core.ComputeQueue().queue->ExecuteCommandLists(1, lists);
    m_core.ComputeQueue().CpuWait(m_core.ComputeQueue().Signal());
}

// Dispatch a one-group-per-sample kernel, chunked under the 65535 group limit.
static void DispatchPerSample(ID3D12GraphicsCommandList* list, ID3D12PipelineState* pso,
                              uint32_t S, uint32_t layer, uint32_t flags) {
    for (uint32_t base = 0; base < S; base += kMaxGroups) {
        uint32_t groups = (S - base < kMaxGroups) ? (S - base) : kMaxGroups;
        uint32_t c[4] = { layer, 0, flags, base };
        list->SetComputeRoot32BitConstants(1, 4, c, 0);
        list->SetPipelineState(pso);
        list->Dispatch(groups, 1, 1);
    }
}

void GpuTrainer::InitWeights(const float* params, uint64_t count) {
    // Two upload buffers: the init params, and a zero block reused for both Adam
    // moment buffers. All copies execute from one command list.
    auto makeUpload = [&](uint64_t bytes, ComPtr<ID3D12Resource>& out, const void* src) {
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc = {1, 0}; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        TIF(m_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&out)), "init upload");
        void* p = nullptr; D3D12_RANGE nr{0, 0};
        out->Map(0, &nr, &p);
        if (src) memcpy(p, src, bytes); else memset(p, 0, bytes);
        out->Unmap(0, nullptr);
    };
    ComPtr<ID3D12Resource> upParams, upZero;
    makeUpload(count * 4, upParams, params);
    makeUpload(m_P * 4, upZero, nullptr);

    ID3D12Resource* weights = m_core.Buffers().weights[0].res.Get();
    Begin();
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = { weights, 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COPY_DEST };
    m_list->ResourceBarrier(1, &b);
    m_list->CopyBufferRegion(weights, 0, upParams.Get(), 0, count * 4);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_list->ResourceBarrier(1, &b);

    // Adam moments: UAV -> COPY_DEST, zero, -> UAV.
    D3D12_RESOURCE_BARRIER am[2]{};
    for (int i = 0; i < 2; ++i) {
        am[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        am[i].Transition = { i ? m_adamV.res.Get() : m_adamM.res.Get(), 0,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST };
    }
    m_list->ResourceBarrier(2, am);
    m_list->CopyBufferRegion(m_adamM.res.Get(), 0, upZero.Get(), 0, m_P * 4);
    m_list->CopyBufferRegion(m_adamV.res.Get(), 0, upZero.Get(), 0, m_P * 4);
    for (int i = 0; i < 2; ++i) std::swap(am[i].Transition.StateBefore, am[i].Transition.StateAfter);
    m_list->ResourceBarrier(2, am);
    ExecuteAndWait();   // upload buffers must outlive submission — they do (scoped here)
    m_adamT = 0;
}

float GpuTrainer::TrainStep(uint32_t epochs) {
    ID3D12Resource* weights = m_core.Buffers().weights[0].res.Get();
    if (weights != m_boundWeights) WriteDescriptors(weights);

    Begin();
    auto wbar = [&](D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = { weights, 0, from, to };
        m_list->ResourceBarrier(1, &b);
    };
    wbar(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    SetPass(0, 0, 0); Dispatch(m_pso.zeroStats.Get(), 1); UavBarrier();
    SetPass(0, 0, 0); Dispatch(m_pso.gae.Get(), (m_N + 63) / 64); UavBarrier();
    SetPass(0, 0, 0); Dispatch(m_pso.advNorm.Get(), (m_S + 63) / 64); UavBarrier();

    // Old log-probs under the rollout policy (current weights), once.
    DispatchPerSample(m_list.Get(), m_pso.forward.Get(), m_S, 0, /*writeOld*/1);
    UavBarrier();

    const uint32_t Ms[3] = { HD, HD, m_Aout };
    const uint32_t Ks[3] = { m_O, HD, HD };
    for (uint32_t e = 0; e < epochs; ++e) {
        ++m_adamT;
        DispatchPerSample(m_list.Get(), m_pso.forward.Get(), m_S, 0, 0);  UavBarrier();
        DispatchPerSample(m_list.Get(), m_pso.backward.Get(), m_S, 0, 0); UavBarrier();
        for (uint32_t L = 0; L < 3; ++L) {
            SetPass(L, 0, 0);
            Dispatch(m_pso.weightGrad.Get(), (Ms[L] + 15) / 16, (Ks[L] + 15) / 16);
            SetPass(L, 0, 0);
            Dispatch(m_pso.biasGrad.Get(), (Ms[L] + 63) / 64);
        }
        UavBarrier();
        SetPass(0, m_adamT, 0); Dispatch(m_pso.adam.Get(),
            static_cast<uint32_t>((m_P + 255) / 256));
        UavBarrier();
    }

    wbar(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Copy reward stat for the UI (Stats must be readable as COPY_SOURCE first).
    D3D12_RESOURCE_BARRIER sb{};
    sb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    sb.Transition = { m_stats.res.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_COPY_SOURCE };
    m_list->ResourceBarrier(1, &sb);
    m_list->CopyBufferRegion(m_statsReadback.Get(), 0, m_stats.res.Get(), 0, 8 * sizeof(uint32_t));
    std::swap(sb.Transition.StateBefore, sb.Transition.StateAfter);
    m_list->ResourceBarrier(1, &sb);

    ExecuteAndWait();

    uint32_t* st = nullptr;
    D3D12_RANGE rr{0, 8 * sizeof(uint32_t)};
    m_statsReadback->Map(0, &rr, reinterpret_cast<void**>(&st));
    float rewSum = st ? *reinterpret_cast<float*>(&st[2]) : 0.f;
    D3D12_RANGE nw{0, 0}; m_statsReadback->Unmap(0, &nw);
    return rewSum / float(m_H * m_N);   // mean reward over the horizon
}

void GpuTrainer::DownloadWeights(float* out, uint64_t count) {
    ID3D12Resource* weights = m_core.Buffers().weights[0].res.Get();
    Begin();
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = { weights, 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COPY_SOURCE };
    m_list->ResourceBarrier(1, &b);
    m_list->CopyBufferRegion(m_weightsReadback.Get(), 0, weights, 0, count * 4);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_list->ResourceBarrier(1, &b);
    ExecuteAndWait();
    float* p = nullptr; D3D12_RANGE rr{0, count * 4};
    m_weightsReadback->Map(0, &rr, reinterpret_cast<void**>(&p));
    if (p) memcpy(out, p, count * 4);
    D3D12_RANGE nw{0, 0}; m_weightsReadback->Unmap(0, &nw);
}

void GpuTrainer::RunGradientOnly() {
    ID3D12Resource* weights = m_core.Buffers().weights[0].res.Get();
    if (weights != m_boundWeights) WriteDescriptors(weights);
    Begin();
    D3D12_RESOURCE_BARRIER wb{};
    wb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    wb.Transition = { weights, 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS };
    m_list->ResourceBarrier(1, &wb);

    SetPass(0, 0, 0); Dispatch(m_pso.zeroStats.Get(), 1); UavBarrier();
    SetPass(0, 0, 0); Dispatch(m_pso.gae.Get(), (m_N + 63) / 64); UavBarrier();
    SetPass(0, 0, 0); Dispatch(m_pso.advNorm.Get(), (m_S + 63) / 64); UavBarrier();
    DispatchPerSample(m_list.Get(), m_pso.forward.Get(), m_S, 0, 1); UavBarrier();
    DispatchPerSample(m_list.Get(), m_pso.forward.Get(), m_S, 0, 0); UavBarrier();
    DispatchPerSample(m_list.Get(), m_pso.backward.Get(), m_S, 0, 0); UavBarrier();
    const uint32_t Ms[3] = { HD, HD, m_Aout };
    const uint32_t Ks[3] = { m_O, HD, HD };
    for (uint32_t L = 0; L < 3; ++L) {
        SetPass(L, 0, 0); Dispatch(m_pso.weightGrad.Get(), (Ms[L] + 15) / 16, (Ks[L] + 15) / 16);
        SetPass(L, 0, 0); Dispatch(m_pso.biasGrad.Get(), (Ms[L] + 63) / 64);
    }
    UavBarrier();
    std::swap(wb.Transition.StateBefore, wb.Transition.StateAfter);
    m_list->ResourceBarrier(1, &wb);
    ExecuteAndWait();
}

void GpuTrainer::DownloadGrad(float* out, uint64_t count) {
    Begin();
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = { m_grad.res.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE };
    m_list->ResourceBarrier(1, &b);
    m_list->CopyBufferRegion(m_gradReadback.Get(), 0, m_grad.res.Get(), 0, count * 4);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_list->ResourceBarrier(1, &b);
    ExecuteAndWait();
    float* p = nullptr; D3D12_RANGE rr{0, count * 4};
    m_gradReadback->Map(0, &rr, reinterpret_cast<void**>(&p));
    if (p) memcpy(out, p, count * 4);
    D3D12_RANGE nw{0, 0}; m_gradReadback->Unmap(0, &nw);
}

} // namespace aistank
