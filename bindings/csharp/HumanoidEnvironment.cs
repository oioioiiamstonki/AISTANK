// HumanoidEnvironment.cs — safe high-level wrapper over the native engine.
// Exposes rollouts as Span<float> views directly over the pinned native readback
// memory: no copies cross the interop boundary.
using System;
using Aistank.Interop;

namespace Aistank;

public sealed class HumanoidEnvironment : IDisposable
{
    private IntPtr _handle;

    public uint NumAgents { get; }
    public uint ObservationDim { get; }
    public uint ActionDim { get; }
    public ulong PolicyParamCount { get; }

    public HumanoidEnvironment(EnvironmentOptions options)
    {
        var cfg = new AistankConfig
        {
            NumAgents = options.NumAgents,
            PhysicsSubsteps = options.PhysicsSubsteps,
            RolloutHorizon = options.RolloutHorizon,
            CpuWorkers = options.CpuWorkers,
            EnablePreview = options.EnablePreview ? 1u : 0u,
            ControlDt = options.ControlDt,
            WForward = options.RewardWeights.Forward,
            WAlive = options.RewardWeights.Alive,
            WEnergy = options.RewardWeights.Energy,
            WUpright = options.RewardWeights.Upright,
            TargetVelocity = options.TargetVelocity,
            TerminationHeight = options.TerminationHeight,
        };
        var result = Native.EngineCreate(in cfg, options.MjcfPath, out _handle);
        if (result != AistankResult.Ok)
            throw new InvalidOperationException($"Engine_Create failed: {result}");

        NumAgents = options.NumAgents;
        ObservationDim = Native.EngineGetObservationDim(_handle);
        ActionDim = Native.EngineGetActionDim(_handle);
        PolicyParamCount = Native.EngineGetPolicyParamCount(_handle);
    }

    /// <summary>One control step across all agents. ~zero CPU cost beyond physics.</summary>
    public AistankStepStats Tick()
    {
        ThrowIfDisposed();
        var result = Native.EngineTick(_handle, out var stats);
        if (result != AistankResult.Ok)
            throw new InvalidOperationException($"Engine_Tick failed: {result} — {LastError()}");
        return stats;
    }

    /// <summary>Pushes new FP32 policy parameters; takes effect next tick (double-buffered).</summary>
    public unsafe void SetPolicyWeights(ReadOnlySpan<float> weights)
    {
        ThrowIfDisposed();
        if ((ulong)weights.Length != PolicyParamCount)
            throw new ArgumentException($"Expected {PolicyParamCount} params, got {weights.Length}");
        fixed (float* p = weights)
        {
            var result = Native.EngineSetPolicyWeights(_handle, p, (ulong)weights.Length);
            if (result != AistankResult.Ok)
                throw new InvalidOperationException($"SetPolicyWeights failed: {result}");
        }
    }

    /// <summary>
    /// Blocks until the latest horizon's GPU readback completes, then exposes the
    /// rollout as spans over pinned native memory. Layout: [step][component][agent]
    /// (SoA). Views are valid until the next Tick().
    /// </summary>
    public unsafe RolloutView MapRollout()
    {
        ThrowIfDisposed();
        var result = Native.EngineMapRollout(_handle,
            out float* obs, out float* act, out float* rew, out byte* done,
            out float* val, out uint horizon, out uint obsDim, out uint actDim);
        if (result != AistankResult.Ok)
            throw new InvalidOperationException($"MapRollout failed: {result}");

        int n = (int)NumAgents, h = (int)horizon;
        return new RolloutView(
            new ReadOnlySpan<float>(obs, h * (int)obsDim * n),
            new ReadOnlySpan<float>(act, h * (int)actDim * n),
            new ReadOnlySpan<float>(rew, h * n),
            new ReadOnlySpan<byte>(done, h * n * sizeof(float)),
            new ReadOnlySpan<float>(val, h * n),
            h, (int)obsDim, (int)actDim, n);
    }

    /// <summary>Number of rigid bodies in the humanoid model (incl. world body 0).</summary>
    public uint BodyCount => Native.EngineGetBodyCount(_handle);

    /// <summary>Parent body index for each body (0 = world). Length = BodyCount.</summary>
    public unsafe int[] GetBodyParents()
    {
        ThrowIfDisposed();
        var parents = new int[BodyCount];
        fixed (int* p = parents) Native.EngineGetBodyParents(_handle, p);
        return parents;
    }

    /// <summary>
    /// World-space body positions (xyz triplets) of one agent, read straight from
    /// its mjData. Safe to call from any thread for visualization — a frame read
    /// concurrently with a physics step may tear, which only glitches one frame.
    /// </summary>
    public unsafe bool TryGetAgentBodyPositions(uint agent, float[] xyz)
    {
        if (_handle == IntPtr.Zero || xyz.Length < BodyCount * 3) return false;
        fixed (float* p = xyz)
            return Native.EngineGetAgentBodyPositions(_handle, agent, p) == AistankResult.Ok;
    }

    // ---- GPU-resident PPO training -----------------------------------------

    /// <summary>Allocates the GPU trainer (buffers, PSOs). Call once before GPU training.</summary>
    public void InitGpuTrainer()
    {
        ThrowIfDisposed();
        if (Native.EngineInitGpuTrainer(_handle) != AistankResult.Ok)
            throw new InvalidOperationException($"InitGpuTrainer failed — {LastError()}");
    }

    /// <summary>Uploads initial policy weights into the GPU's slot 0 and zeroes Adam state.</summary>
    public unsafe void InitPolicyWeights(ReadOnlySpan<float> weights)
    {
        ThrowIfDisposed();
        fixed (float* p = weights)
            if (Native.EngineInitPolicyWeights(_handle, p, (ulong)weights.Length) != AistankResult.Ok)
                throw new InvalidOperationException($"InitPolicyWeights failed — {LastError()}");
    }

    /// <summary>Runs one PPO update entirely on the GPU. Returns mean reward over the horizon.</summary>
    public float TrainStepGpu(uint epochs = 3)
    {
        ThrowIfDisposed();
        if (Native.EngineTrainStepGpu(_handle, epochs, out float r) != AistankResult.Ok)
            throw new InvalidOperationException($"TrainStepGpu failed — {LastError()}");
        return r;
    }

    /// <summary>Copies the current GPU policy weights to CPU.</summary>
    public unsafe void DownloadWeights(Span<float> dst)
    {
        ThrowIfDisposed();
        fixed (float* p = dst) Native.EngineDownloadWeights(_handle, p, (ulong)dst.Length);
    }

    /// <summary>Parity test: one GPU forward/backward, then read the summed gradient.</summary>
    public unsafe void GpuGradient(Span<float> grad)
    {
        ThrowIfDisposed();
        Native.EngineRunGradientOnly(_handle);
        fixed (float* p = grad) Native.EngineDownloadGrad(_handle, p, (ulong)grad.Length);
    }

    /// <summary>Number of collision geoms in the model (includes the ground plane).</summary>
    public uint GeomCount => Native.EngineGetGeomCount(_handle);

    /// <summary>Static geom description: MuJoCo type enum + size[3] per geom.</summary>
    public unsafe (int[] Types, float[] Sizes) GetGeomStatic()
    {
        ThrowIfDisposed();
        int g = (int)GeomCount;
        var types = new int[g];
        var sizes = new float[g * 3];
        fixed (int* t = types) fixed (float* s = sizes)
            Native.EngineGetGeomStatic(_handle, t, s);
        return (types, sizes);
    }

    /// <summary>
    /// One agent's per-geom world transforms this frame: positions (geom*3) and
    /// row-major 3x3 rotations (geom*9), read straight from mjData. Concurrent
    /// reads with a physics step may tear, glitching at most one frame.
    /// </summary>
    public unsafe bool TryGetAgentGeomPose(uint agent, float[] xpos, float[] xmat)
    {
        if (_handle == IntPtr.Zero || xpos.Length < GeomCount * 3 || xmat.Length < GeomCount * 9)
            return false;
        fixed (float* p = xpos) fixed (float* mtx = xmat)
            return Native.EngineGetAgentGeomPose(_handle, agent, p, mtx) == AistankResult.Ok;
    }

    private unsafe string LastError()
        => System.Runtime.InteropServices.Marshal.PtrToStringUTF8(
               Native.EngineGetLastError(_handle)) ?? "";

    private void ThrowIfDisposed()
    {
        if (_handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(HumanoidEnvironment));
    }

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            Native.EngineDestroy(_handle);
            _handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    ~HumanoidEnvironment() => Dispose();
}

public readonly ref struct RolloutView
{
    public readonly ReadOnlySpan<float> Observations;
    public readonly ReadOnlySpan<float> Actions;
    public readonly ReadOnlySpan<float> Rewards;
    public readonly ReadOnlySpan<byte> Dones;     // float-encoded 0/1, raw bytes
    public readonly ReadOnlySpan<float> Values;
    public readonly int Horizon, ObsDim, ActDim, NumAgents;

    public RolloutView(ReadOnlySpan<float> obs, ReadOnlySpan<float> act,
                       ReadOnlySpan<float> rew, ReadOnlySpan<byte> done,
                       ReadOnlySpan<float> val, int horizon, int obsDim,
                       int actDim, int numAgents)
    {
        Observations = obs; Actions = act; Rewards = rew; Dones = done; Values = val;
        Horizon = horizon; ObsDim = obsDim; ActDim = actDim; NumAgents = numAgents;
    }
}

public sealed record EnvironmentOptions
{
    public required string MjcfPath { get; init; }
    public uint NumAgents { get; init; } = 4096;
    public uint PhysicsSubsteps { get; init; } = 4;       // 240 Hz physics @ 60 Hz control
    public uint RolloutHorizon { get; init; } = 32;
    public uint CpuWorkers { get; init; } = 0;            // 0 = all cores
    public bool EnablePreview { get; init; } = false;
    public float ControlDt { get; init; } = 1f / 60f;
    public float TargetVelocity { get; init; } = 1.5f;
    public float TerminationHeight { get; init; } = 0.55f;
    public RewardWeights RewardWeights { get; init; } = new();
}

public sealed record RewardWeights
{
    public float Forward { get; init; } = 1.25f;
    public float Alive { get; init; } = 0.5f;
    public float Energy { get; init; } = 0.005f;
    public float Upright { get; init; } = 0.3f;
}
