// GpuTrainer.h — GPU-resident PPO update (no CPU readback of rollouts).
//
// Runs GAE → forward → backward → tiled weight-gradient reduction → Adam as a
// chain of compute dispatches over the rollout buffers that SimBuffers already
// fills each horizon. Owns its own root signature, descriptor heap, PSOs, and
// scratch buffers; shares the device, compute queue, weights, and rollout
// buffers with EngineCore.
#pragma once
#include "GpuBuffers.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>

namespace aistank {

using Microsoft::WRL::ComPtr;
class EngineCore;

// 16-byte-aligned mirror of cbuffer TrainConstants in TrainShaders.hlsl.
struct alignas(256) TrainConstantsCB {
    uint32_t S, O, A, Aout, N, H, P;
    uint32_t OffW1, OffB1, OffW2, OffB2, OffW3, OffB3;
    float Gamma, Lambda, ClipEps, ValueCoef, Sigma, Lr, Beta1, Beta2, AdamEps;
    uint32_t _pad;
};

class GpuTrainer {
public:
    explicit GpuTrainer(EngineCore& core);   // throws std::runtime_error on failure

    // Uploads initial policy weights to slot 0 and zeroes the Adam moments.
    void InitWeights(const float* params, uint64_t count);

    // One PPO update over the current rollout, writing the active weight slot in
    // place. Blocks until the GPU finishes. Returns mean reward over the horizon.
    float TrainStep(uint32_t epochs);

    // Copies the active policy weights to CPU (for checkpoints / parity tests).
    void DownloadWeights(float* out, uint64_t count);

    // Copies the summed (un-averaged) gradient buffer to CPU after a single
    // forward/backward pass — used only by the gradient-parity self-test.
    void RunGradientOnly();
    void DownloadGrad(float* out, uint64_t count);

    uint32_t SampleCount() const { return m_S; }

private:
    void CreateBuffers();
    void CreateRootSigAndPipelines();
    void WriteDescriptors(ID3D12Resource* weights);
    void Begin();                                   // reset list, set heap/root/cbv
    void SetPass(uint32_t layer, uint32_t adamT, uint32_t flags);
    void Dispatch(ID3D12PipelineState* pso, uint32_t gx, uint32_t gy = 1);
    void UavBarrier();
    void ExecuteAndWait();

    EngineCore& m_core;
    ID3D12Device* m_dev = nullptr;

    uint32_t m_S = 0, m_O = 0, m_A = 0, m_Aout = 0, m_N = 0, m_H = 0;
    uint64_t m_P = 0;

    ComPtr<ID3D12RootSignature>  m_rootSig;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT m_inc = 0;

    struct Pso { ComPtr<ID3D12PipelineState> gae, advNorm, forward, backward,
                                             weightGrad, biasGrad, adam, zeroStats; } m_pso;

    // Scratch buffers (default heap, UAV).
    GpuBuffer m_grad, m_adamM, m_adamV;
    GpuBuffer m_adv, m_vtarget, m_oldLogp;
    GpuBuffer m_xbuf, m_h1, m_h2, m_dz1, m_dz2, m_dz3;
    GpuBuffer m_stats;

    ComPtr<ID3D12Resource> m_constants;             // upload heap, persistently mapped
    TrainConstantsCB*      m_constMapped = nullptr;

    ComPtr<ID3D12CommandAllocator>    m_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_list;

    ComPtr<ID3D12Resource> m_statsReadback;         // 8 uints
    ComPtr<ID3D12Resource> m_weightsReadback;       // P floats
    ComPtr<ID3D12Resource> m_gradReadback;          // P floats

    ID3D12Resource* m_boundWeights = nullptr;       // current u0 target
    bool m_adamInit = false;
    uint32_t m_adamT = 0;
};

} // namespace aistank
