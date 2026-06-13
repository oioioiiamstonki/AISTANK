"""
train_arena.py — a crowd of color-coded humanoids learning to walk together,
fully on the GPU (MuJoCo physics via MJX + JAX PPO), then rendered to a video.

One MuJoCo world holds N humanoids that don't collide with each other (see
arena.py). Each humanoid is an independent RL agent sharing one policy; with
per-agent exploration noise and resets they diverge, so you watch a whole crowd
of differently-colored robots train at once. For training throughput we also
vmap E copies of the arena (E*N agents); only arena 0 is rendered.

Everything that is "training" runs on the GPU:
  physics (mjx.step) · policy inference · GAE · PPO update (Adam) — all jitted.

Run in WSL2/Linux with CUDA:
  ~/mjxenv/bin/python mjx/train_arena.py --agents 16 --arenas 32 --updates 300 \
       --render mjx/arena.gif
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

import arena as arena_mod
from train_humanoid import (
    GAMMA, LAMBDA, CLIP, VALUE_COEF, LR, W_FWD, W_ALIVE, W_CTRL, W_UPRIGHT,
    TARGET_VEL, TERM_HEIGHT, quat_rotate, init_policy, policy_forward, log_prob,
)

QPOS_PER = 23   # 7 free-joint + 16 hinge
QVEL_PER = 22   # 6 free-joint + 16 hinge


def humanoid_obs(qp, qv, prev_action):
    quat = qp[3:7]
    qinv = jp.array([quat[0], -quat[1], -quat[2], -quat[3]])
    lin = quat_rotate(qinv, qv[0:3])
    ang = quat_rotate(qinv, qv[3:6])
    return jp.concatenate([qp[2:3], quat, qp[7:23], lin, ang, qv[6:22] * 0.1, prev_action])


def humanoid_reward(qp, qv, action):
    vx = qv[0]
    up = quat_rotate(qp[3:7], jp.array([0.0, 0.0, 1.0]))[2]
    r = (W_FWD * jp.clip(vx, 0.0, TARGET_VEL) + W_ALIVE
         - W_CTRL * jp.sum(action ** 2) - W_UPRIGHT * (1.0 - up))
    fell = (qp[2] < TERM_HEIGHT) | (up < 0.3)
    return r - jp.where(fell, 1.0, 0.0), fell


class Arena:
    """One MJX world of N non-colliding humanoids; each humanoid is an agent."""

    def __init__(self, n_agents, n_substeps=5):
        self.n = n_agents
        self.mj = mujoco.MjModel.from_xml_string(arena_mod.build_arena_xml(n_agents))
        self.mx = mjx.put_model(self.mj)
        self.n_substeps = n_substeps
        self.qpos0 = jp.array(self.mj.qpos0).reshape(n_agents, QPOS_PER)
        self.obs_dim = int(humanoid_obs(self.qpos0[0], jp.zeros(QVEL_PER), jp.zeros(16)).shape[0])

    def obs(self, dx, prev_act):                       # (n, obs_dim)
        qp = dx.qpos.reshape(self.n, QPOS_PER)
        qv = dx.qvel.reshape(self.n, QVEL_PER)
        return jax.vmap(humanoid_obs)(qp, qv, prev_act)

    def reward(self, dx, action):                      # (n,), (n,)
        qp = dx.qpos.reshape(self.n, QPOS_PER)
        qv = dx.qvel.reshape(self.n, QVEL_PER)
        return jax.vmap(humanoid_reward)(qp, qv, action)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        dx = mjx.make_data(self.mx)
        kp, kv = jax.random.split(key)
        jitter = self.qpos0.at[:, 7:].add(0.01 * jax.random.normal(kp, (self.n, 16)))
        jitter = jitter.at[:, 2].add(0.01 * jax.random.uniform(kv, (self.n,)))
        dx = dx.replace(qpos=jitter.reshape(-1))
        return mjx.forward(self.mx, dx)

    @partial(jax.jit, static_argnums=0)
    def step(self, dx, action):                        # action (n,16)
        ctrl = jp.clip(action, -1.0, 1.0).reshape(-1)

        def body(d, _):
            return mjx.step(self.mx, d.replace(ctrl=ctrl)), None

        dx, _ = jax.lax.scan(body, dx, None, length=self.n_substeps)
        return dx

    def reset_done(self, dx, done, action, key):
        """Reset only the humanoids that fell (no mjx.forward — obs reads qpos/qvel)."""
        qp = dx.qpos.reshape(self.n, QPOS_PER)
        qv = dx.qvel.reshape(self.n, QVEL_PER)
        rqp = self.qpos0.at[:, 7:].add(0.01 * jax.random.normal(key, (self.n, 16)))
        m = done[:, None]
        qp = jp.where(m, rqp, qp)
        qv = jp.where(m, jp.zeros_like(qv), qv)
        dx = dx.replace(qpos=qp.reshape(-1), qvel=qv.reshape(-1))
        nact = jp.where(done[:, None], 0.0, action)
        return dx, self.obs(dx, nact), nact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=16, help="humanoids per arena (all rendered)")
    ap.add_argument("--arenas", type=int, default=32, help="vmapped arena copies for throughput")
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", type=str, default="mjx/arena.gif")
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    env = Arena(args.agents)
    E, N, T = args.arenas, args.agents, args.rollout
    print(f"arena: {N} colored humanoids x {E} copies = {E*N} agents | "
          f"obs={env.obs_dim} nu_total={env.mj.nu}")

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    data = jax.vmap(env.reset)(jax.random.split(sub, E))
    prev = jp.zeros((E, N, 16))
    obs = jax.vmap(env.obs)(data, prev)

    key, sub = jax.random.split(key)
    params = init_policy(sub, env.obs_dim, 16)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    vstep = jax.vmap(env.step)
    vobs = jax.vmap(env.obs)
    vrew = jax.vmap(env.reward)
    vreset_done = jax.vmap(env.reset_done)

    @jax.jit
    def rollout(params, data, obs, prev, key):
        def stepfn(carry, _):
            data, obs, prev, key = carry
            mu, v, log_std = policy_forward(params, obs)           # (E,N,16),(E,N)
            key, k = jax.random.split(key)
            action = jp.clip(mu + jax.random.normal(k, mu.shape) * jp.exp(log_std), -1, 1)
            lp = log_prob(mu, log_std, action)
            data = vstep(data, action)
            rew, done = vrew(data, action)
            key, k = jax.random.split(key)
            data, nobs, nprev = vreset_done(data, done, action, jax.random.split(k, E))
            return (data, nobs, nprev, key), (obs, action, lp, v, rew, done)

        carry, traj = jax.lax.scan(stepfn, (data, obs, prev, key), None, length=T)
        data, obs, prev, key = carry
        return traj, policy_forward(params, obs)[1], data, obs, prev, key

    def gae(rew, val, done, last_v):
        def bwd(c, x):
            g, nv = c
            r, v, d = x
            nd = 1.0 - d
            delta = r + GAMMA * nv * nd - v
            g = delta + GAMMA * LAMBDA * nd * g
            return (g, v), g
        _, adv = jax.lax.scan(bwd, (jp.zeros_like(last_v), last_v), (rew, val, done), reverse=True)
        return adv, adv + val

    @jax.jit
    def update(params, opt_state, traj, last_v, key):
        obs, action, old_lp, val, rew, done = traj
        adv, vtgt = gae(rew, val, done, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape((-1,) + x.shape[3:])
        obs, action, old_lp, adv, vtgt = map(flat, (obs, action, old_lp, adv, vtgt))

        def loss_fn(p, ob, ac, olp, ad, vt):
            mu, v, log_std = policy_forward(p, ob)
            ratio = jp.exp(log_prob(mu, log_std, ac) - olp)
            pg = -jp.mean(jp.minimum(ratio * ad, jp.clip(ratio, 1 - CLIP, 1 + CLIP) * ad))
            return pg + VALUE_COEF * jp.mean((v - vt) ** 2)

        B = obs.shape[0]; mb = max(B // 4, 1)
        def epoch(c, _):
            params, opt_state, key = c
            key, k = jax.random.split(key)
            idx = jax.random.permutation(k, B)[: (B // mb) * mb].reshape(-1, mb)
            def mbatch(c, ii):
                params, opt_state = c
                g = jax.grad(loss_fn)(params, obs[ii], action[ii], old_lp[ii], adv[ii], vtgt[ii])
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), None
            (params, opt_state), _ = jax.lax.scan(mbatch, (params, opt_state), idx)
            return (params, opt_state, key), None
        (params, opt_state, key), _ = jax.lax.scan(epoch, (params, opt_state, key), None, length=args.epochs)
        return params, opt_state, key, jp.mean(rew)

    t0 = time.time()
    traj, last_v, data, obs, prev, key = rollout(params, data, obs, prev, key)
    params, opt_state, key, mr = update(params, opt_state, traj, last_v, key)
    mr.block_until_ready()
    print(f"compiled in {time.time()-t0:.1f}s")

    t0 = time.time()
    for u in range(args.updates):
        traj, last_v, data, obs, prev, key = rollout(params, data, obs, prev, key)
        params, opt_state, key, mr = update(params, opt_state, traj, last_v, key)
        if u % 20 == 0:
            mr.block_until_ready()
            sps = E * N * T * (u + 1) / (time.time() - t0)
            print(f"update {u:4d} | mean_reward {float(mr):7.3f} | {sps/1e6:5.2f}M env-steps/s")

    if args.render:
        render_rollout(env, params, args.render, key)
    print("done.")


def render_rollout(env, params, out_path, key, steps=240):
    """Roll out the trained policy on one arena and render the colored crowd."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio

    dx = env.reset(jax.random.PRNGKey(123))
    obs = env.obs(dx, jp.zeros((env.n, 16)))
    step = jax.jit(env.step)
    qframes = []
    for _ in range(steps):
        mu, _, _ = policy_forward(params, obs)          # deterministic (no noise)
        action = jp.clip(mu, -1, 1)
        dx = step(dx, action)
        rew, done = env.reward(dx, action)
        dx, obs, _ = env.reset_done(dx, done, action, jax.random.PRNGKey(0))
        qframes.append(np.array(dx.qpos))

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.2 * env.n ** 0.5
    cam.elevation = -18
    cam.azimuth = 110
    cam.lookat[:] = [0, 0, 0.7]
    d = mujoco.MjData(env.mj)
    frames = []
    with mujoco.Renderer(env.mj, 720, 1280) as r:
        for qp in qframes:
            d.qpos[:] = qp
            mujoco.mj_forward(env.mj, d)
            r.update_scene(d, cam)
            frames.append(r.render())
    imageio.mimsave(out_path, frames, fps=30)
    Image_save = out_path.rsplit(".", 1)[0] + "_frame.png"
    imageio.imwrite(Image_save, frames[len(frames) // 2])
    print(f"wrote {out_path} ({len(frames)} frames) and {Image_save}")


if __name__ == "__main__":
    main()
