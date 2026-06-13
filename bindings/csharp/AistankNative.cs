// AistankNative.cs — zero-cost P/Invoke surface over include/aistank.h.
// All structs are blittable; LibraryImport generates marshal-free stubs at compile time.
using System;
using System.Runtime.InteropServices;

namespace Aistank.Interop;

public enum AistankResult : int
{
    Ok = 0,
    DeviceError = 1,
    MujocoError = 2,
    BadArgument = 3,
    OutOfMemory = 4,
}

[StructLayout(LayoutKind.Sequential)]
public struct AistankConfig
{
    public uint NumAgents;
    public uint PhysicsSubsteps;
    public uint RolloutHorizon;
    public uint CpuWorkers;
    public uint EnablePreview;
    public uint PreviewHwndLo;
    public uint PreviewHwndHi;
    public float ControlDt;
    public float WForward, WAlive, WEnergy, WUpright;
    public float TargetVelocity;
    public float TerminationHeight;
}

[StructLayout(LayoutKind.Sequential)]
public struct AistankStepStats
{
    public ulong Tick;
    public float MeanReward;
    public float MeanEpisodeLen;
    public uint ResetsThisTick;
    public float CpuPhysicsMs;
    public float GpuComputeMs;
}

public static unsafe partial class Native
{
    private const string Dll = "aistank";

    [LibraryImport(Dll, EntryPoint = "Engine_Create", StringMarshalling = StringMarshalling.Utf8)]
    public static partial AistankResult EngineCreate(in AistankConfig cfg, string mjcfPath, out IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_Destroy")]
    public static partial void EngineDestroy(IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_Tick")]
    public static partial AistankResult EngineTick(IntPtr engine, out AistankStepStats stats);

    [LibraryImport(Dll, EntryPoint = "Engine_SetPolicyWeights")]
    public static partial AistankResult EngineSetPolicyWeights(IntPtr engine, float* @params, ulong count);

    [LibraryImport(Dll, EntryPoint = "Engine_GetPolicyParamCount")]
    public static partial ulong EngineGetPolicyParamCount(IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_MapRollout")]
    public static partial AistankResult EngineMapRollout(IntPtr engine,
        out float* obs, out float* actions, out float* rewards, out byte* dones,
        out float* values, out uint horizon, out uint obsDim, out uint actDim);

    [LibraryImport(Dll, EntryPoint = "Engine_GetBodyCount")]
    public static partial uint EngineGetBodyCount(IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_GetBodyParents")]
    public static partial AistankResult EngineGetBodyParents(IntPtr engine, int* outParents);

    [LibraryImport(Dll, EntryPoint = "Engine_GetAgentBodyPositions")]
    public static partial AistankResult EngineGetAgentBodyPositions(IntPtr engine, uint agent, float* outXyz);

    [LibraryImport(Dll, EntryPoint = "Engine_GetObservationDim")]
    public static partial uint EngineGetObservationDim(IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_GetActionDim")]
    public static partial uint EngineGetActionDim(IntPtr engine);

    [LibraryImport(Dll, EntryPoint = "Engine_GetLastError")]
    public static partial IntPtr EngineGetLastError(IntPtr engine);
}
