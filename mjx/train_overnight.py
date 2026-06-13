"""
train_overnight.py — robust, long-running GPU training for humanoid walking.

Built for unattended overnight runs:
  • observation normalization (running mean/std) — the key ingredient that lets
    PPO actually converge to a walking gait rather than just balancing
  • periodic checkpointing + --resume (a crash or Ctrl-C never loses progress)
  • logging to mjx/runs/<timestamp>/train.log (tail it in the morning)
  • periodic render snapshots of the color-coded crowd so you can see it improve
  • Ctrl-C saves a final checkpoint before exiting

Throughput: training runs on thousands of *independent* single-humanoid MJX
envs (fast — small per-env constraint system), which is ~40x quicker than
stepping one giant many-humanoid world. The learned per-humanoid policy is then
rendered on a multi-humanoid arena so you still get the colored-crowd visual.

Everything that is training runs on the GPU: MuJoCo physics (MJX), inference,
GAE, and the PPO/Adam update.

Overnight launch (WSL2 + CUDA), detached:
    cd /mnt/c/Users/adria/Desktop/AISTANK
    MUJOCO_GL=egl nohup ~/mjxenv/bin/python mjx/train_overnight.py \
        --envs 4096 --updates 20000 --resume > /dev/null 2>&1 &
    tail -f mjx/runs/latest/train.log          # watch progress
    ~/mjxenv/bin/python mjx/train_overnight.py --play mjx/checkpoints/walk.pkl
"""
import argparse
import datetime
import os
import pickle
import signal
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax

from train_humanoid import (
    GAMMA, LAMBDA, CLIP, VALUE_COEF, LR, MjxEnv, make_obs, reward_and_done,
    init_policy, policy_forward, log_prob,
)
from train_arena import Arena

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"


# ----------------------------------------------------------------- obs norm

def norm_apply(obs, mean, std):
    return jp.clip((obs - mean) / std, -10.0, 10.0)


def norm_update(mean, m2, count, batch):
    """Parallel (Chan) variance merge of a batch of observations [B, O]."""
    b_n = batch.shape[0]
    b_mean = batch.mean(0)
    b_m2 = ((batch - b_mean) ** 2).sum(0)
    delta = b_mean - mean
    new_count = count + b_n
    new_mean = mean + delta * (b_n / new_count)
    new_m2 = m2 + b_m2 + delta ** 2 * (count * b_n / new_count)
    return new_mean, new_m2, new_count


def nforward(params, obs, mean, std):
    return policy_forward(params, norm_apply(obs, mean, std))


# ----------------------------------------------------------------- checkpoint

def save_ckpt(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(state), f)
    os.replace(tmp, path)        # atomic


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------- training

def build_steps(env, N, T, epochs):
    vstep, vreset, vobs, vrew = (jax.vmap(env.step_one), jax.vmap(env.reset_one),
                                 jax.vmap(make_obs), jax.vmap(reward_and_done))

    @jax.jit
    def rollout(params, mean, std, data, obs, prev, key):
        def stepfn(carry, _):
            data, obs, prev, key = carry
            mu, v, log_std = nforward(params, obs, mean, std)
            key, k = jax.random.split(key)
            action = jp.clip(mu + jax.random.normal(k, mu.shape) * jp.exp(log_std), -1, 1)
            lp = log_prob(mu, log_std, action)
            data = vstep(data, action)
            rew, done = vrew(data, action)
            key, k = jax.random.split(key)
            rdata = vreset(jax.random.split(k, obs.shape[0]))
            data = jax.tree.map(
                lambda a, b: jp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a),
                data, rdata)
            nact = jp.where(done[:, None], 0.0, action)
            nobs = vobs(data, nact)
            return (data, nobs, nact, key), (obs, action, lp, v, rew, done)

        carry, traj = jax.lax.scan(stepfn, (data, obs, prev, key), None, length=T)
        data, obs, prev, key = carry
        last_v = nforward(params, obs, mean, std)[1]
        return traj, last_v, data, obs, prev, key

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
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--render-every", type=int, default=500)
    ap.add_argument("--render-agents", type=int, default=16)
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "walk.pkl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--play", type=str, default="", help="render a checkpoint to a gif and exit")
    args = ap.parse_args()

    if args.play:
        st = load_ckpt(Path(args.play))
        _play(Arena(args.render_agents), st, ROOT / "play.gif")
        return

    env = MjxEnv()
    N, T = args.envs, args.rollout

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = RUNS_DIR / ts
    run.mkdir(parents=True, exist_ok=True)
    latest = RUNS_DIR / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run.name)
    except OSError:
        pass
    logf = open(run / "train.log", "a", buffering=1)

    def log(msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        print(line); logf.write(line + "\n")

    log(f"JAX backend: {jax.default_backend()} devices: {jax.devices()}")
    log(f"{N} envs | obs={env.obs_dim} act={env.nu} | rollout={T} epochs={args.epochs}")

    key = jax.random.PRNGKey(args.seed)
    key, sub = jax.random.split(key)
    params = init_policy(sub, env.obs_dim, env.nu)
    rollout, update, opt = build_steps(env, N, T, args.epochs)
    opt_state = opt.init(params)
    mean = jp.zeros(env.obs_dim); count = jp.array(1e-4)
    m2 = jp.full(env.obs_dim, 1e-4); start = 0

    if args.resume and Path(args.ckpt).exists():
        st = load_ckpt(Path(args.ckpt))
        params, opt_state = st["params"], st["opt_state"]
        mean, m2, count = st["mean"], st["m2"], st["count"]
        start = st["update"]
        log(f"resumed from {args.ckpt} at update {start}")

    key, sub = jax.random.split(key)
    data = jax.vmap(env.reset_one)(jax.random.split(sub, N))
    prev = jp.zeros((N, env.nu))
    obs = jax.vmap(make_obs)(data, prev)

    render_env = Arena(args.render_agents) if args.render_every else None

    def snapshot(u):
        save_ckpt(Path(args.ckpt), dict(params=params, opt_state=opt_state,
                                        mean=mean, m2=m2, count=count, update=u))

    interrupted = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__("v", True))

    t0 = time.time(); best = -1e9; u = start
    for u in range(start, args.updates):
        std = jp.sqrt(m2 / count) + 1e-5
        traj, last_v, data, obs, prev, key = rollout(params, mean, std, data, obs, prev, key)
        params, opt_state, key, mr, flat_obs = update(
            params, opt_state, mean, std, traj, last_v, key)
        mean, m2, count = norm_update(mean, m2, count, flat_obs)

        if u % 10 == 0:
            mr = float(mr)
            sps = N * T * (u - start + 1) / (time.time() - t0)
            log(f"update {u:6d} | mean_reward {mr:8.3f} | {sps/1e6:6.2f}M env-steps/s")
            best = max(best, mr)
        if u > start and u % args.save_every == 0:
            snapshot(u); log(f"  checkpoint @ {u}")
        if render_env is not None and u > start and u % args.render_every == 0:
            try:
                _render(render_env, params, mean, m2, count, run / f"frame_{u:06d}.png")
                log(f"  render @ {u}  ->  runs/{ts}/frame_{u:06d}.png")
            except Exception as e:
                log(f"  render failed: {e}")
        if interrupted["v"]:
            log("interrupted — saving and exiting"); break

    snapshot(u + 1)
    log(f"done. best mean_reward {best:.3f}. checkpoint: {args.ckpt}")
    logf.close()


# ----------------------------------------------------------------- rendering

def _rollout_crowd(arena, params, mean, std, steps):
    dx = arena.reset(jax.random.PRNGKey(7))
    obs = arena.obs(dx, jp.zeros((arena.n, 16)))
    step = jax.jit(arena.step)
    qframes = []
    for _ in range(steps):
        mu, _, _ = nforward(params, obs, mean, std)
        action = jp.clip(mu, -1, 1)
        dx = step(dx, action)
        _, done = arena.reward(dx, action)
        dx, obs, _ = arena.reset_done(dx, done, action, jax.random.PRNGKey(0))
        qframes.append(np.array(dx.qpos))
    return qframes


def _camera(arena):
    import mujoco
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.2 * arena.n ** 0.5; cam.elevation = -18; cam.azimuth = 110
    cam.lookat[:] = [0, 0, 0.7]
    return cam


def _render(arena, params, mean, m2, count, out_png):
    import mujoco, imageio
    os.environ.setdefault("MUJOCO_GL", "egl")
    std = jp.sqrt(m2 / count) + 1e-5
    qp = _rollout_crowd(arena, params, mean, std, 120)[-1]
    d = mujoco.MjData(arena.mj); d.qpos[:] = qp; mujoco.mj_forward(arena.mj, d)
    with mujoco.Renderer(arena.mj, 720, 1280) as r:
        r.update_scene(d, _camera(arena)); imageio.imwrite(str(out_png), r.render())


def _play(arena, st, out_gif, steps=300):
    import mujoco, imageio
    os.environ.setdefault("MUJOCO_GL", "egl")
    std = jp.sqrt(st["m2"] / st["count"]) + 1e-5
    qframes = _rollout_crowd(arena, st["params"], st["mean"], std, steps)
    d = mujoco.MjData(arena.mj); cam = _camera(arena); frames = []
    with mujoco.Renderer(arena.mj, 720, 1280) as r:
        for qp in qframes:
            d.qpos[:] = qp; mujoco.mj_forward(arena.mj, d)
            r.update_scene(d, cam); frames.append(r.render())
    imageio.mimsave(str(out_gif), frames, fps=30)
    print(f"wrote {out_gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
