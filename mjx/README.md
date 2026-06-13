# GPU MuJoCo physics — the MJX training path

This is the **"real MuJoCo on the GPU"** path. It uses [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html),
MuJoCo's official JAX/XLA reimplementation, so the *physics solver itself* runs on
the GPU — not just the RL update. The entire loop is GPU-resident:

| Stage | Device |
|---|---|
| Physics (`mjx.step`, MuJoCo on XLA) | **GPU** |
| Policy inference (JAX MLP) | **GPU** |
| PPO update (GAE + clipped surrogate + Adam) | **GPU** |

It trains the same 16-DoF humanoid ([`../assets/humanoid16.xml`](../assets/humanoid16.xml))
as the rest of the repo to walk.

> **This is a separate stack from the Windows DX12 editor.** The C++/DX12 engine
> (`src/`, `editor/`) uses the CPU MuJoCo C API for physics and custom HLSL for the
> RL update. MJX requires JAX, which only does GPU on **Linux or WSL2 + CUDA** —
> *not* native Windows. The two paths share only the MJCF humanoid asset.

## Setup (WSL2 / Linux, NVIDIA GPU)

```bash
# Inside WSL2 Ubuntu (or native Linux) with an NVIDIA driver visible to the GPU:
python3 -m venv ~/mjxenv
~/mjxenv/bin/pip install -U pip
~/mjxenv/bin/pip install -r mjx/requirements.txt

# sanity check — should print a CudaDevice and "gpu"
~/mjxenv/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## Train

```bash
cd /path/to/AISTANK
~/mjxenv/bin/python mjx/train_humanoid.py --envs 2048 --updates 200 --save mjx/policy.pkl
```

Output (verified on an RTX 5080 in WSL2):

```
JAX backend: gpu  devices: [CudaDevice(id=0)]
compiled in 38.7s; obs_dim=59 act=16
update    0 | mean_reward   0.448 | ...
update   20 | mean_reward   0.753 | ...
update   40 | mean_reward   1.207 | ...
update   50 | mean_reward   1.231 | ...
```

Rising mean reward = the humanoid is learning to stay upright and move forward.
Walking gaits need longer runs (thousands of updates); this scaffold establishes
the correct, fully-GPU pipeline.

## Notes

- First call pays a one-time XLA compilation cost (~30–60 s) for the whole
  rollout+update graph; subsequent updates are fast.
- The reward, observation layout, and termination conditions mirror the DX12
  engine's `CS_RewardAndTerminate` so the two implementations are comparable.
- `--save` writes the policy (a pickled JAX pytree) for later evaluation/rendering.
