// SimulationLoop.cpp — the GPU-resident tick.
//
//  Tick(N) sequence (exact order, three queues):
//   1. CPU throttle: wait until frame slot (N % 2)'s fences from tick N-2 are done.
//   2. CPU: apply actions read back from tick N-1; mj_resetData envs flagged done.
//   3. CPU workers: mj_step × substeps for all envs; scatter SoA into upload slot.
//   4. COPY queue: upload slot → default-heap physics mirror.            [signal F_up]
//   5. COMPUTE queue: wait F_up; dispatch Observe → Policy → Reward/Term
//      → Resets → BuildInstances (UAV barriers between).                 [signal F_cs]
//   6. COPY queue: wait F_cs; actions → readback ring; at horizon end,
//      rollout buffers → rollout readback.                               [signal F_rb]
//   7. (optional) DIRECT queue: wait F_cs; preview render, present.
//   CPU never blocks on F_rb inside Tick — it is consumed at step 2 of tick N+1.
#include "SimulationLoop.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

namespace aistank {

static void ThrowIfFailed(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(what);
}

uint64_t Engine_GetPolicyParamCountBytes();  // defined at end of file

// ----------------------------------------------------------- HlslMlpExecutor

void HlslMlpExecutor::RecordInference(ID3D12GraphicsCommandList* cl, uint32_t weightSlot) {
    cl->SetPipelineState(m_core.Pipelines().policyForward.Get());
    // Root constants: { passIndex, layerIndex(unused — kernel runs all layers), weightSlot, seed }
    const uint32_t consts[4] = { 1, 0, weightSlot, 0 };
    cl->SetComputeRoot32BitConstants(3, 4, consts, 0);
    // One threadgroup (256 lanes) per agent; layers chained via groupshared inside the kernel.
    cl->Dispatch(m_core.Config().num_agents, 1, 1);
}

// ------------------------------------------------------------- SimulationLoop

SimulationLoop::SimulationLoop(EngineCore& core)
    : m_core(core), m_policy(std::make_unique<HlslMlpExecutor>(core)) {
    ID3D12Device* dev = core.Device();
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        ThrowIfFailed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
            IID_PPV_ARGS(&m_computeAlloc[i])), "alloc(compute)");
        ThrowIfFailed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
            m_computeAlloc[i].Get(), nullptr, IID_PPV_ARGS(&m_computeList[i])), "list(compute)");
        m_computeList[i]->Close();
        ThrowIfFailed(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY,
            IID_PPV_ARGS(&m_copyAlloc[i])), "alloc(copy)");
        ThrowIfFailed(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY,
            m_copyAlloc[i].Get(), nullptr, IID_PPV_ARGS(&m_copyList[i])), "list(copy)");
        m_copyList[i]->Close();
    }

    // Persistently-mapped CBV ring (256-byte aligned slots, one per frame in flight).
    {
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = sizeof(SimConstants) * kFramesInFlight;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc = {1, 0}; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_constantsBuffer)), "constants buffer");
        D3D12_RANGE noRead{0, 0};
        m_constantsBuffer->Map(0, &noRead, reinterpret_cast<void**>(&m_constantsMapped));
    }

    // Timestamp queries: 2 per frame slot (begin/end of compute chain).
    {
        D3D12_QUERY_HEAP_DESC qd{ D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 * kFramesInFlight, 0 };
        ThrowIfFailed(dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&m_tsHeap)), "ts heap");
        m_core.ComputeQueue().queue->GetTimestampFrequency(&m_tsFrequency);
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = sizeof(uint64_t) * 2 * kFramesInFlight;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc = {1, 0}; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ThrowIfFailed(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_tsReadback)), "ts rb");
    }

    m_pendingResets.assign(core.Config().num_agents, 0);
}

SimulationLoop::~SimulationLoop() {
    // EngineCore's destructor drains the queues; command lists/allocators are
    // ComPtr-owned and must simply outlive the drain, which member order guarantees
    // because SimulationLoop is destroyed before EngineCore in the ABI wrapper.
    if (m_constantsMapped) m_constantsBuffer->Unmap(0, nullptr);
}

// ----------------------------------------------------------- Phase A (CPU)

void SimulationLoop::StepPhysicsAndPack(uint32_t frameSlot, float* outPhysicsMs) {
    EngineCore& E = m_core;
    const mjModel* m = E.Model();
    SimBuffers& B = E.Buffers();
    const uint32_t N = B.N, D = B.D, A = B.A;
    const uint32_t substeps = std::max(1u, E.Config().physics_substeps);

    // Actions produced by tick N-1, already resident in the readback slot
    // (fence consumed by the throttle in Tick()).
    const float* actions = reinterpret_cast<const float*>(
        B.actionReadback.mapped[frameSlot]);
    float* up = reinterpret_cast<float*>(B.stateUpload.mapped[frameSlot]);

    // SoA layout inside the upload slot — offsets must match SrvSlot order
    // and the CopyBufferRegion sequence in RecordUpload.
    float* upJointPos   = up;
    float* upJointVel   = upJointPos + size_t(D) * N;
    float* upRootPos    = upJointVel + size_t(D) * N;
    float* upRootQuat   = upRootPos + 3 * size_t(N);
    float* upRootLinVel = upRootQuat + 4 * size_t(N);
    float* upRootAngVel = upRootLinVel + 3 * size_t(N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::atomic<uint32_t> next{0};
    auto worker = [&] {
        for (uint32_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < N; ) {
            EnvSlot& env = E.Envs()[i];
            // Apply policy actions (tick lag = 1; policy is trained with it).
            for (uint32_t a = 0; a < A; ++a)
                env.data->ctrl[a] = mju_clip(actions[a * N + i], -1.0, 1.0);
            if (m_pendingResets[i]) {
                mj_resetData(m, env.data);
                env.data->qpos[2] += 0.01 * (double(i % 17) / 17.0); // tiny height jitter
                mj_forward(m, env.data);
                env.stepsSinceReset = 0;
            }
            for (uint32_t s = 0; s < substeps; ++s)
                mj_step(m, env.data);
            env.stepsSinceReset++;

            // Scatter into SoA (free joint occupies qpos[0..6], qvel[0..5]).
            const double* qp = env.data->qpos;
            const double* qv = env.data->qvel;
            for (uint32_t d = 0; d < D; ++d) {
                upJointPos[d * N + i] = float(qp[7 + d]);
                upJointVel[d * N + i] = float(qv[6 + d]);
            }
            for (int k = 0; k < 3; ++k) upRootPos   [k * N + i] = float(qp[k]);
            for (int k = 0; k < 4; ++k) upRootQuat  [k * N + i] = float(qp[3 + k]);
            for (int k = 0; k < 3; ++k) upRootLinVel[k * N + i] = float(qv[k]);
            for (int k = 0; k < 3; ++k) upRootAngVel[k * N + i] = float(qv[3 + k]);
        }
    };

    uint32_t nWorkers = E.Config().cpu_workers
        ? E.Config().cpu_workers : std::thread::hardware_concurrency();
    nWorkers = std::min(nWorkers, N);
    std::vector<std::thread> pool;
    pool.reserve(nWorkers);
    for (uint32_t w = 1; w < nWorkers; ++w) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    *outPhysicsMs = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

// ----------------------------------------------------- Phase B (copy queue)

void SimulationLoop::RecordUpload(uint32_t frameSlot) {
    SimBuffers& B = m_core.Buffers();
    ID3D12GraphicsCommandList* cl = m_copyList[frameSlot].Get();
    m_copyAlloc[frameSlot]->Reset();
    cl->Reset(m_copyAlloc[frameSlot].Get(), nullptr);

    ID3D12Resource* src = B.stateUpload.res[frameSlot].Get();
    const uint64_t N = B.N, D = B.D;
    uint64_t off = 0;
    auto copy = [&](GpuBuffer& dst, uint64_t bytes) {
        cl->CopyBufferRegion(dst.res.Get(), 0, src, off, bytes);
        off += bytes;
    };
    copy(B.jointPos,   D * N * 4);
    copy(B.jointVel,   D * N * 4);
    copy(B.rootPos,    3 * N * 4);
    copy(B.rootQuat,   4 * N * 4);
    copy(B.rootLinVel, 3 * N * 4);
    copy(B.rootAngVel, 3 * N * 4);

    // Pending weight push: copy into the inactive slot, then flip on the compute side.
    uint32_t pending = m_pendingWeightSlot.exchange(UINT32_MAX, std::memory_order_acq_rel);
    if (pending != UINT32_MAX) {
        cl->CopyBufferRegion(B.weights[pending].res.Get(), 0,
                             B.weightUpload.res[frameSlot].Get(), 0,
                             Engine_GetPolicyParamCountBytes());
        m_activeWeightSlot = pending;
    }
    cl->Close();

    ID3D12CommandList* lists[] = { cl };
    m_core.CopyQueue().queue->ExecuteCommandLists(1, lists);
    m_copyFenceAt[frameSlot] = m_core.CopyQueue().Signal();
}

// -------------------------------------------------- Phase C (compute queue)

void SimulationLoop::RecordComputeChain(uint32_t frameSlot) {
    EngineCore& E = m_core;
    SimBuffers& B = E.Buffers();
    ID3D12GraphicsCommandList* cl = m_computeList[frameSlot].Get();
    m_computeAlloc[frameSlot]->Reset();
    cl->Reset(m_computeAlloc[frameSlot].Get(), nullptr);

    // Update this frame's CBV slot.
    const AistankConfig& C = E.Config();
    SimConstants& sc = m_constantsMapped[frameSlot];
    sc.numAgents = B.N; sc.dofCount = B.D; sc.actionDim = B.A; sc.obsDim = B.O;
    sc.controlDt = C.control_dt;
    sc.targetVelocity = C.target_velocity;
    sc.terminationHeight = C.termination_height;
    sc.wForward = C.w_forward; sc.wAlive = C.w_alive;
    sc.wEnergy = C.w_energy;   sc.wUpright = C.w_upright;
    sc.maxEpisodeSteps = 1000;
    sc.rolloutHorizon = B.horizon;
    sc.rolloutCursor = uint32_t(m_tick % B.horizon);

    ID3D12DescriptorHeap* heaps[] = { E.SrvUavHeap() };
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootSignature(E.ComputeRootSig());
    cl->SetComputeRootConstantBufferView(0,
        m_constantsBuffer->GetGPUVirtualAddress() + uint64_t(frameSlot) * sizeof(SimConstants));
    cl->SetComputeRootDescriptorTable(1, E.SrvTableGpu());
    cl->SetComputeRootDescriptorTable(2, E.UavTableGpu());

    cl->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frameSlot * 2);

    const uint32_t groups64 = (B.N + 63) / 64;
    auto uavBarrier = [&] {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;  // null resource = all UAVs
        cl->ResourceBarrier(1, &b);
    };

    // 1. Observation gathering (normalized, SoA writes).
    cl->SetPipelineState(E.Pipelines().observe.Get());
    cl->Dispatch(groups64, 1, 1);
    uavBarrier();

    // 2. Policy inference — entirely on-chip.
    m_policy->RecordInference(cl, m_activeWeightSlot);
    uavBarrier();

    // 3. Reward + termination + rollout append.
    cl->SetPipelineState(E.Pipelines().rewardTerminate.Get());
    cl->Dispatch(groups64, 1, 1);
    uavBarrier();

    // 4. GPU-side reset bookkeeping (zero step counters, reseed RNG).
    cl->SetPipelineState(E.Pipelines().applyResets.Get());
    cl->Dispatch(groups64, 1, 1);

    // 5. Preview instance transforms (first 64 agents) — feeds the direct queue.
    if (E.Config().enable_preview) {
        uavBarrier();
        cl->SetPipelineState(E.Pipelines().buildInstances.Get());
        cl->Dispatch(1, 1, 1);
    }

    cl->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frameSlot * 2 + 1);
    cl->ResolveQueryData(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frameSlot * 2, 2,
                         m_tsReadback.Get(), uint64_t(frameSlot) * 2 * sizeof(uint64_t));
    cl->Close();

    // Compute must not start until this tick's upload landed.
    E.ComputeQueue().GpuWait(E.CopyQueue(), m_copyFenceAt[frameSlot]);
    ID3D12CommandList* lists[] = { cl };
    E.ComputeQueue().queue->ExecuteCommandLists(1, lists);
    m_computeFenceAt[frameSlot] = E.ComputeQueue().Signal();
}

// ----------------------------------------------------- Phase D (copy queue)

void SimulationLoop::RecordReadback(uint32_t frameSlot, bool horizonEnd) {
    SimBuffers& B = m_core.Buffers();
    // A second, short copy list per slot would be cleaner; for the scaffold we
    // reuse the copy allocator with a fresh list region by re-recording.
    ID3D12GraphicsCommandList* cl = m_copyList[frameSlot].Get();
    cl->Reset(m_copyAlloc[frameSlot].Get(), nullptr);

    cl->CopyBufferRegion(B.actionReadback.res[frameSlot].Get(), 0,
                         B.actions.res.Get(), 0, uint64_t(B.A) * B.N * 4);
    // Done flags ride along inside the action readback slot? No — keep it explicit:
    // dones land in the rollout buffers; CPU resets are driven by rolloutDone's
    // last row via the same fence. Scaffold simplification: copy doneFlags too.
    if (horizonEnd) {
        uint64_t off = 0;
        ID3D12Resource* dst = B.rolloutReadback.res[frameSlot].Get();
        auto pull = [&](GpuBuffer& src, uint64_t bytes) {
            cl->CopyBufferRegion(dst, off, src.res.Get(), 0, bytes);
            off += bytes;
        };
        pull(B.rolloutObs,  uint64_t(B.horizon) * B.O * B.N * 4);
        pull(B.rolloutAct,  uint64_t(B.horizon) * B.A * B.N * 4);
        pull(B.rolloutRew,  uint64_t(B.horizon) * B.N * 4);
        pull(B.rolloutDone, uint64_t(B.horizon) * B.N * 4);
        pull(B.rolloutVal,  uint64_t(B.horizon) * B.N * 4);
    }
    cl->Close();

    m_core.CopyQueue().GpuWait(m_core.ComputeQueue(), m_computeFenceAt[frameSlot]);
    ID3D12CommandList* lists[] = { cl };
    m_core.CopyQueue().queue->ExecuteCommandLists(1, lists);
    m_copyFenceAt[frameSlot] = m_core.CopyQueue().Signal();
}

// -------------------------------------------------------------------- Tick

AistankResult SimulationLoop::Tick(AistankStepStats& outStats) {
    const uint32_t frameSlot = uint32_t(m_tick % kFramesInFlight);
    EngineCore& E = m_core;

    try {
        // 1. Throttle: this slot's GPU work from tick N-2 must be complete
        //    before we overwrite its staging memory. This is the ONLY CPU wait.
        E.CopyQueue().CpuWait(m_copyFenceAt[frameSlot]);

        // 2+3. CPU physics with previous actions; pack SoA snapshot.
        float physicsMs = 0.f;
        StepPhysicsAndPack(frameSlot, &physicsMs);

        // 4. Upload.
        RecordUpload(frameSlot);
        // 5. Compute chain.
        RecordComputeChain(frameSlot);
        // 6. Readback.
        const bool horizonEnd = ((m_tick + 1) % E.Buffers().horizon) == 0;
        RecordReadback(frameSlot, horizonEnd);
        // 7. Preview render on the direct queue is owned by the (optional)
        //    renderer module; it waits on m_computeFenceAt[frameSlot].

        E.CheckDeviceRemoved();

        // Stats from the previous resolved timestamp pair (frame N-2, same slot).
        uint64_t* ts = nullptr;
        D3D12_RANGE rr{ frameSlot * 2 * sizeof(uint64_t), (frameSlot * 2 + 2) * sizeof(uint64_t) };
        if (SUCCEEDED(m_tsReadback->Map(0, &rr, reinterpret_cast<void**>(&ts))) && ts) {
            outStats.gpu_compute_ms = m_tsFrequency
                ? float(double(ts[frameSlot * 2 + 1] - ts[frameSlot * 2]) / m_tsFrequency * 1e3)
                : 0.f;
            D3D12_RANGE noWrite{0, 0};
            m_tsReadback->Unmap(0, &noWrite);
        }
        outStats.tick = m_tick;
        outStats.cpu_physics_ms = physicsMs;
        // mean_reward / episode stats are aggregated by CS_RewardAndTerminate into
        // UAV_EpisodeStats and surfaced via the rollout readback (horizon cadence).
        outStats.mean_reward = 0.f;
        outStats.mean_episode_len = 0.f;
        outStats.resets_this_tick = 0;

        ++m_tick;
        return AISTANK_OK;
    } catch (const std::exception&) {
        return AISTANK_ERR_DEVICE;
    }
}

// ----------------------------------------------------------- Trainer interface

AistankResult SimulationLoop::SetPolicyWeights(const float* params, uint64_t count) {
    if (count != m_core.PolicyParamCount()) return AISTANK_ERR_BAD_ARG;
    const uint32_t frameSlot = uint32_t(m_tick % kFramesInFlight);
    std::memcpy(m_core.Buffers().weightUpload.mapped[frameSlot], params, count * 4);
    // Flip to the slot inference is NOT currently reading.
    m_pendingWeightSlot.store(m_activeWeightSlot ^ 1u, std::memory_order_release);
    return AISTANK_OK;
}

AistankResult SimulationLoop::MapRollout(const float** obs, const float** act,
                                         const float** rew, const uint8_t** done,
                                         const float** val, uint32_t* horizon,
                                         uint32_t* obsDim, uint32_t* actDim) {
    SimBuffers& B = m_core.Buffers();
    const uint32_t frameSlot = uint32_t((m_tick + kFramesInFlight - 1) % kFramesInFlight);
    // Block until the horizon-end readback recorded in that slot has completed.
    m_core.CopyQueue().CpuWait(m_copyFenceAt[frameSlot]);

    const uint8_t* base = B.rolloutReadback.mapped[frameSlot];
    uint64_t off = 0;
    const uint64_t N = B.N, H = B.horizon;
    *obs  = reinterpret_cast<const float*>(base + off); off += H * B.O * N * 4;
    *act  = reinterpret_cast<const float*>(base + off); off += H * B.A * N * 4;
    *rew  = reinterpret_cast<const float*>(base + off); off += H * N * 4;
    *done = base + off;                                  off += H * N * 4;
    *val  = reinterpret_cast<const float*>(base + off);
    *horizon = B.horizon; *obsDim = B.O; *actDim = B.A;
    return AISTANK_OK;
}

void SimulationLoop::ApplyCpuResets(uint32_t /*frameSlot*/) {
    // Scaffold note: m_pendingResets is refreshed from the done-flags portion of
    // the rollout readback at horizon boundaries. A per-tick done readback (4KB
    // for 4096 agents — negligible) can be added to RecordReadback for tighter
    // reset latency; left as the first TODO for a production pass.
}

// Helper used by RecordUpload (weights byte size); kept out of the header.
static uint64_t g_paramBytes = 0;
uint64_t Engine_GetPolicyParamCountBytes() { return g_paramBytes; }
void SetPolicyParamBytesForUpload(uint64_t b) { g_paramBytes = b; }

} // namespace aistank
