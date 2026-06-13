"""
train_humanoid.py — GPU-resident humanoid walking with real MuJoCo physics (MJX).

Everything runs on the GPU through JAX/XLA:
  • physics            — MuJoCo MJX (mjx.step), batched over thousands of envs
  • policy inference   — JAX MLP
  • PPO update         — GAE + clipped surrogate + Adam (optax), all jitted

This is the "real MuJoCo on the GPU" path: MJX *is* MuJoCo, reimplemented on XLA.
It is a separate stack from the C++/DX12 engine in the rest of the repo (which
uses the CPU MuJoCo C API for physics and custom HLSL for the RL update).

Run inside WSL2/Linux with a CUDA GPU:
    ~/mjxenv/bin/python mjx/train_humanoid.py --envs 2048 --updates 200
"""
import argparse
import time
from functools import partial

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import optax
from mujoco import mjx

# ----------------------------------------------------------------- config

ASSET = "assets/humanoid16.xml"
HIDDEN = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.0
LR = 3e-4
LOG_STD_INIT = -1.6  # ~ln(0.2), matches the DX12 engine's exploration sigma

# Reward weights (mirror the DX12 engine's CS_RewardAndTerminate).
W_FWD, W_ALIVE, W_CTRL, W_UPRIGHT = 1.25, 0.5, 0.005, 0.3
TARGET_VEL = 1.5
TERM_HEIGHT = 0.55


# ----------------------------------------------------------------- env (MJX)

def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    u = jp.array([x, y, z])
    return v + 2.0 * jp.cross(u, jp.cross(u, v) + w * v)


def make_obs(dx, prev_action):
    qpos, qvel = dx.qpos, dx.qvel
    quat = qpos[3:7]
    qinv = jp.array([quat[0], -quat[1], -quat[2], -quat[3]])
    lin_local = quat_rotate(qinv, qvel[0:3])
    ang_local = quat_rotate(qinv, qvel[3:6])
    return jp.concatenate([
        qpos[2:3],          # root height
        quat,               # orientation (4)
        qpos[7:],           # joint angles (16)
        lin_local,          # torso-local linear vel (3)
        ang_local,          # torso-local angular vel (3)
        qvel[6:] * 0.1,     # joint velocities, scaled (16)
        prev_action,        # previous action (16)
    ])


def reward_and_done(dx, action):
    vx = dx.qvel[0]                                    # world forward velocity
    up = quat_rotate(dx.qpos[3:7], jp.array([0.0, 0.0, 1.0]))[2]
    rooth = dx.qpos[2]
    r = (W_FWD * jp.clip(vx, 0.0, TARGET_VEL)
         + W_ALIVE
         - W_CTRL * jp.sum(action ** 2)
         - W_UPRIGHT * (1.0 - up))
    fell = (rooth < TERM_HEIGHT) | (up < 0.3)
    r = r - jp.where(fell, 1.0, 0.0)                   # terminal fall penalty
    return r, fell


class MjxEnv:
    """Batched MJX humanoid with automatic per-env reset on termination."""

    def __init__(self, n_substeps=5):
        self.mj = mujoco.MjModel.from_xml_path(ASSET)
        self.mx = mjx.put_model(self.mj)
        self.n_substeps = n_substeps
        self.nu = self.mj.nu
        self.qpos0 = jp.array(self.mj.qpos0)
        self.obs_dim = int(make_obs(self._fresh(), jp.zeros(self.nu)).shape[0])

    def _fresh(self):
        return mjx.make_data(self.mx)

    @partial(jax.jit, static_argnums=0)
    def reset_one(self, key):
        kq, kv = jax.random.split(key)
        dx = mjx.make_data(self.mx)
        qpos = self.qpos0.at[2].add(0.01 * jax.random.uniform(kq))
        qpos = qpos.at[7:].add(0.01 * jax.random.normal(kq, (self.mx.nq - 7,)))
        dx = dx.replace(qpos=qpos, qvel=0.01 * jax.random.normal(kv, (self.mx.nv,)))
        dx = mjx.forward(self.mx, dx)
        return dx

    @partial(jax.jit, static_argnums=0)
    def step_one(self, dx, action):
        ctrl = jp.clip(action, -1.0, 1.0)

        def body(d, _):
            return mjx.step(self.mx, d.replace(ctrl=ctrl)), None

        dx, _ = jax.lax.scan(body, dx, None, length=self.n_substeps)
        return dx


# ----------------------------------------------------------------- policy

def init_policy(key, obs_dim, act_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    def dense(k, fan_in, fan_out, scale):
        w = jax.random.normal(k, (fan_in, fan_out)) * (scale / np.sqrt(fan_in))
        return w, jp.zeros(fan_out)
    return {
        "w1": dense(k1, obs_dim, HIDDEN, np.sqrt(2)),
        "w2": dense(k2, HIDDEN, HIDDEN, np.sqrt(2)),
        "mu": dense(k3, HIDDEN, act_dim, 0.01),
        "v":  dense(k3, HIDDEN, 1, 1.0),
        "log_std": jp.full((act_dim,), LOG_STD_INIT),
    }


def policy_forward(p, obs):
    h = jax.nn.elu(obs @ p["w1"][0] + p["w1"][1])
    h = jax.nn.elu(h @ p["w2"][0] + p["w2"][1])
    mu = jp.tanh(h @ p["mu"][0] + p["mu"][1])
    v = (h @ p["v"][0] + p["v"][1])[..., 0]
    return mu, v, p["log_std"]


def log_prob(mu, log_std, action):
    std = jp.exp(log_std)
    return jp.sum(-0.5 * ((action - mu) / std) ** 2 - log_std - 0.9189385, axis=-1)


# ----------------------------------------------------------------- PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="", help="path to write trained policy (.pkl)")
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    env = MjxEnv()
    N, T = args.envs, args.rollout
    key = jax.random.PRNGKey(args.seed)

    # Batched env state.
    key, sub = jax.random.split(key)
    reset_keys = jax.random.split(sub, N)
    data = jax.vmap(env.reset_one)(reset_keys)
    prev_act = jp.zeros((N, env.nu))
    obs = jax.vmap(make_obs)(data, prev_act)

    key, sub = jax.random.split(key)
    params = init_policy(sub, env.obs_dim, env.nu)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    vstep = jax.vmap(env.step_one)
    vreset = jax.vmap(env.reset_one)
    vobs = jax.vmap(make_obs)
    vrew = jax.vmap(reward_and_done)

    @jax.jit
    def rollout(params, data, obs, prev_act, key):
        """Collect T steps across N envs; auto-reset terminated envs."""
        def stepfn(carry, _):
            data, obs, prev_act, key = carry
            mu, v, log_std = policy_forward(params, obs)
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, mu.shape) * jp.exp(log_std)
            action = jp.clip(mu + noise, -1.0, 1.0)
            lp = log_prob(mu, log_std, action)

            data = vstep(data, action)
            rew, done = vrew(data, action)
            nobs = vobs(data, action)

            # Reset finished envs.
            key, sub = jax.random.split(key)
            rkeys = jax.random.split(sub, obs.shape[0])
            rdata = vreset(rkeys)
            data = jax.tree.map(lambda a, b: jp.where(
                done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), data, rdata)
            nact = jp.where(done[:, None], 0.0, action)
            nobs = jp.where(done[:, None], vobs(rdata, jp.zeros_like(action)), nobs)

            out = (obs, action, lp, v, rew, done)
            return (data, nobs, nact, key), out

        carry, traj = jax.lax.scan(stepfn, (data, obs, prev_act, key), None, length=T)
        data, obs, prev_act, key = carry
        last_v = policy_forward(params, obs)[1]   # bootstrap value for GAE
        return traj, last_v, data, obs, prev_act, key

    def gae(rew, val, done, last_v):
        def bwd(carry, x):
            gae_, next_v = carry
            r, v, d = x
            nd = 1.0 - d
            delta = r + GAMMA * next_v * nd - v
            gae_ = delta + GAMMA * LAMBDA * nd * gae_
            return (gae_, v), gae_
        (_, _), adv = jax.lax.scan(
            bwd, (jp.zeros_like(last_v), last_v),
            (rew, val, done), reverse=True)
        return adv, adv + val

    @jax.jit
    def update(params, opt_state, traj, last_v, key):
        obs, action, old_lp, val, rew, done = traj
        adv, vtarget = gae(rew, val, done, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Flatten (T, N, ...) -> (T*N, ...).
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        obs, action, old_lp, adv, vtarget = map(flat, (obs, action, old_lp, adv, vtarget))

        def loss_fn(p, ob, ac, olp, ad, vt):
            mu, v, log_std = policy_forward(p, ob)
            lp = log_prob(mu, log_std, ac)
            ratio = jp.exp(lp - olp)
            unclipped = ratio * ad
            clipped = jp.clip(ratio, 1 - CLIP, 1 + CLIP) * ad
            pg = -jp.mean(jp.minimum(unclipped, clipped))
            vl = VALUE_COEF * jp.mean((v - vt) ** 2)
            ent = ENTROPY_COEF * jp.mean(log_std)
            return pg + vl - ent

        B = obs.shape[0]
        mb = B // 4
        def epoch(carry, _):
            params, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, B)
            def minib(carry, idx):
                params, opt_state = carry
                g = jax.grad(loss_fn)(params, obs[idx], action[idx], old_lp[idx],
                                      adv[idx], vtarget[idx])
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), None
            idxs = perm[: (B // mb) * mb].reshape(-1, mb)
            (params, opt_state), _ = jax.lax.scan(minib, (params, opt_state), idxs)
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=args.epochs)
        return params, opt_state, key, jp.mean(rew)

    # Compile + warm up.
    t0 = time.time()
    traj, last_v, data, obs, prev_act, key = rollout(params, data, obs, prev_act, key)
    params, opt_state, key, mr = update(params, opt_state, traj, last_v, key)
    mr.block_until_ready()
    print(f"compiled in {time.time() - t0:.1f}s; obs_dim={env.obs_dim} act={env.nu}")

    t0 = time.time()
    for u in range(args.updates):
        traj, last_v, data, obs, prev_act, key = rollout(params, data, obs, prev_act, key)
        params, opt_state, key, mr = update(params, opt_state, traj, last_v, key)
        if u % 10 == 0:
            mr.block_until_ready()
            sps = N * T * (u + 1) / (time.time() - t0)
            print(f"update {u:4d} | mean_reward {float(mr):7.3f} | {sps/1e6:5.2f}M env-steps/s")

    if args.save:
        import pickle
        with open(args.save, "wb") as f:
            pickle.dump(jax.device_get(params), f)
        print(f"saved policy to {args.save}")
    print("done.")


if __name__ == "__main__":
    main()
