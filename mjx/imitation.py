"""
imitation.py — Phase 1: teach the humanoid to walk/run FORWARD (fixing the
sideways gait), on the GPU, as a locomotion prior for soccer.

Two ideas combined:
  • Imitation: track a procedural reference walking gait (DeepMimic-style pose
    tracking against a hand-authored leg/arm cycle). True mocap isn't available
    for this custom 16-DoF body, so the "demonstration" is a synthetic reference.
  • Heading-aware task reward: reward velocity ALONG the torso's facing axis
    (not world +x). This is what removes the side-stepping artifact — you only
    get reward for moving the way you face.

The policy is phase-free with the same 59-dim observation as soccer_env, so the
trained weights transfer directly (see train_soccer.py --init).
Reuses the overnight infra: obs-norm, checkpoint/resume, logging.
"""
import argparse, datetime, os, pickle, signal, time
from functools import partial
from pathlib import Path

import jax, jax.numpy as jp, numpy as np, optax

from train_humanoid import (GAMMA, LAMBDA, CLIP, VALUE_COEF, LR, MjxEnv, make_obs,
                            init_policy, policy_forward, log_prob, quat_rotate)
from train_overnight import norm_apply, norm_update, nforward, save_ckpt, load_ckpt

ROOT = Path(__file__).resolve().parent
TERM_HEIGHT = 0.5
TARGET_SPEED = 2.0
CADENCE = 1.4                      # steps per second of the reference gait
# reward weights
W_TRACK, W_FWD, W_HEAD, W_ALIVE, W_UPRIGHT = 0.5, 1.0, 0.1, 0.5, 0.2


def reference_gait(phase):
    """Target joint angles (16,) for qpos[7:23] at cycle phase (radians)."""
    p = phase
    s, sl = jp.sin(p), jp.sin(p + jp.pi)
    q = jp.zeros(16)
    q = q.at[5].set(-0.45 * s)                  # hip_y_r (fore/aft)
    q = q.at[11].set(-0.45 * sl)                # hip_y_l
    q = q.at[7].set(0.4 * (1 - jp.cos(p)))      # knee_r (flex in swing)
    q = q.at[13].set(0.4 * (1 - jp.cos(p + jp.pi)))   # knee_l
    q = q.at[8].set(0.2 * s)                    # ankle_y_r
    q = q.at[14].set(0.2 * sl)                  # ankle_y_l
    q = q.at[0].set(0.5 * sl)                   # shoulder_r (counter-swing)
    q = q.at[1].set(0.5 * s)                    # shoulder_l
    return q


def imitation_reward(qp, qv, phase):
    q = qp[7:23]
    track = jp.exp(-2.0 * jp.mean((q - reference_gait(phase)) ** 2))
    quat = qp[3:7]
    facing = quat_rotate(quat, jp.array([1.0, 0.0, 0.0]))
    fxy = facing[:2] / (jp.linalg.norm(facing[:2]) + 1e-6)
    fwd = jp.dot(qv[0:3][:2], fxy)              # velocity along facing
    up = quat_rotate(quat, jp.array([0.0, 0.0, 1.0]))[2]
    r = (W_TRACK * track + W_FWD * jp.clip(fwd, 0.0, TARGET_SPEED)
         + W_HEAD * facing[0] + W_ALIVE + W_UPRIGHT * up)
    fell = (qp[2] < TERM_HEIGHT) | (up < 0.0)
    return r - jp.where(fell, 1.0, 0.0), fell


def build(env, N, T, epochs, dphase):
    vstep, vreset, vobs = jax.vmap(env.step_one), jax.vmap(env.reset_one), jax.vmap(make_obs)
    vrew = jax.vmap(imitation_reward)

    @jax.jit
    def rollout(params, mean, std, data, obs, phase, key):
        def stepfn(carry, _):
            data, obs, phase, key = carry
            mu, v, log_std = nforward(params, obs, mean, std)
            key, k = jax.random.split(key)
            action = jp.clip(mu + jax.random.normal(k, mu.shape) * jp.exp(log_std), -1, 1)
            lp = log_prob(mu, log_std, action)
            data = vstep(data, action)
            phase = phase + dphase
            rew, done = vrew(jax.vmap(lambda d: d.qpos)(data), jax.vmap(lambda d: d.qvel)(data), phase)
            key, k = jax.random.split(key)
            rdata = vreset(jax.random.split(k, N))
            data = jax.tree.map(
                lambda a, b: jp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), data, rdata)
            key, k = jax.random.split(key)
            phase = jp.where(done, jax.random.uniform(k, (N,)) * 2 * jp.pi, phase)
            nobs = vobs(data, jp.where(done[:, None], 0.0, action))
            return (data, nobs, phase, key), (obs, action, lp, v, rew, done)
        carry, traj = jax.lax.scan(stepfn, (data, obs, phase, key), None, length=T)
        data, obs, phase, key = carry
        return traj, nforward(params, obs, mean, std)[1], data, obs, phase, key

    def gae(rew, val, done, last_v):
        def bwd(c, x):
            g, nv = c; r, vv, d = x; nd = 1.0 - d
            g = (r + GAMMA * nv * nd - vv) + GAMMA * LAMBDA * nd * g
            return (g, vv), g
        _, adv = jax.lax.scan(bwd, (jp.zeros_like(last_v), last_v), (rew, val, done), reverse=True)
        return adv, adv + val

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))

    @jax.jit
    def update(params, opt_state, mean, std, traj, last_v, key):
        obs, action, old_lp, val, rew, done = traj
        adv, vtgt = gae(rew, val, done, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        obs, action, old_lp, adv, vtgt = map(flat, (obs, action, old_lp, adv, vtgt))

        def loss_fn(p, ob, ac, olp, ad, vt):
            mu, v, log_std = nforward(p, ob, mean, std)
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
        (params, opt_state, key), _ = jax.lax.scan(epoch, (params, opt_state, key), None, length=epochs)
        return params, opt_state, key, jp.mean(rew), obs
    return rollout, update, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "walk_fwd.pkl"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    env = MjxEnv()
    N, T = args.envs, args.rollout
    dphase = 2 * jp.pi * CADENCE * env.n_substeps * float(env.mx.opt.timestep)

    run = ROOT / "runs" / datetime.datetime.now().strftime("imit_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a", buffering=1)
    def log(m):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(line); logf.write(line + "\n")

    log(f"JAX {jax.default_backend()} {jax.devices()} | {N} envs obs={env.obs_dim}")
    key = jax.random.PRNGKey(0)
    key, s = jax.random.split(key)
    params = init_policy(s, env.obs_dim, env.nu)
    rollout, update, opt = build(env, N, T, args.epochs, dphase)
    opt_state = opt.init(params)
    mean = jp.zeros(env.obs_dim); m2 = jp.full(env.obs_dim, 1e-4); count = jp.array(1e-4); start = 0
    if args.resume and Path(args.ckpt).exists():
        st = load_ckpt(Path(args.ckpt)); params, opt_state = st["params"], st["opt_state"]
        mean, m2, count, start = st["mean"], st["m2"], st["count"], st["update"]
        log(f"resumed @ {start}")

    key, s = jax.random.split(key)
    data = jax.vmap(env.reset_one)(jax.random.split(s, N))
    phase = jax.random.uniform(s, (N,)) * 2 * jp.pi
    obs = jax.vmap(make_obs)(data, jp.zeros((N, env.nu)))

    def snap(u): save_ckpt(Path(args.ckpt), dict(params=params, opt_state=opt_state,
                                                 mean=mean, m2=m2, count=count, update=u))
    stop = {"v": False}; signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    t0 = time.time(); u = start
    for u in range(start, args.updates):
        std = jp.sqrt(m2 / count) + 1e-5
        traj, last_v, data, obs, phase, key = rollout(params, mean, std, data, obs, phase, key)
        params, opt_state, key, mr, fo = update(params, opt_state, mean, std, traj, last_v, key)
        mean, m2, count = norm_update(mean, m2, count, fo)
        if u % 10 == 0:
            log(f"update {u:6d} | reward {float(mr):7.3f} | {N*T*(u-start+1)/(time.time()-t0)/1e6:5.2f}M/s")
        if u > start and u % args.save_every == 0:
            snap(u); log(f"  ckpt @ {u}")
        if (args.hours and time.time() - t0 >= args.hours * 3600) or stop["v"]:
            log("stopping"); break
    snap(u + 1); log(f"done. ckpt {args.ckpt}"); logf.close()


if __name__ == "__main__":
    main()
