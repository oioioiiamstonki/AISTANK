"""
train_soccer.py — Phase 2: self-play PPO for 2v2 soccer on the GPU.

All four players share one policy (symmetric self-play): every player is a
training sample, both teams improve together. The policy is warm-started from
the Phase-1 walking checkpoint via --init: the 59-dim locomotion block of the
input layer (and all hidden/output layers) are copied, so the players already
know how to walk; the 18 extra soccer inputs start at zero and are learned.

Reward (per player, from soccer_env): stay upright + get to the ball + drive the
ball toward the opponent goal + score. The ball curves (Magnus) and drags
realistically, all on the GPU.

Honest scope: this is a runnable self-play setup that learns ball-directed
behavior. Skillful, coordinated 2v2 passing is a large-compute research problem;
this gives the correct pipeline and a long-training entry point.
"""
import argparse, datetime, os, pickle, signal, time
from functools import partial
from pathlib import Path

import jax, jax.numpy as jp, numpy as np, optax

from train_humanoid import GAMMA, LAMBDA, CLIP, VALUE_COEF, LR, init_policy, policy_forward, log_prob
from train_overnight import nforward, norm_update, save_ckpt, load_ckpt
from soccer_env import SoccerEnv, NP

ROOT = Path(__file__).resolve().parent
LOCO_OBS = 59          # locomotion block shared with the walk policy


def transfer_from_walk(loco_path, obs_dim, act_dim, key):
    """Warm-start a soccer policy from a locomotion checkpoint."""
    fresh = init_policy(key, obs_dim, act_dim)
    if not loco_path or not Path(loco_path).exists():
        return fresh, jp.zeros(obs_dim), jp.full(obs_dim, 1e-4), jp.array(1e-4), False
    st = load_ckpt(Path(loco_path)); lp = st["params"]
    w1 = jp.zeros((obs_dim, 256)).at[:LOCO_OBS].set(lp["w1"][0])      # extras -> 0
    params = {"w1": (w1, lp["w1"][1]), "w2": lp["w2"], "mu": lp["mu"],
              "v": lp["v"], "log_std": lp["log_std"]}
    mean = jp.zeros(obs_dim).at[:LOCO_OBS].set(st["mean"])
    m2 = jp.full(obs_dim, st["count"]).at[:LOCO_OBS].set(st["m2"])
    return params, mean, m2, jp.array(st["count"]), True


def build(env, E, T, epochs):
    vstep, vreset, vobs, vrew = (jax.vmap(env.step), jax.vmap(env.reset),
                                 jax.vmap(env.observe), jax.vmap(env.reward_done))

    @jax.jit
    def rollout(params, mean, std, data, obs, key):
        def stepfn(carry, _):
            data, obs, key = carry
            mu, v, log_std = nforward(params, obs, mean, std)         # (E,4,16),(E,4)
            key, k = jax.random.split(key)
            action = jp.clip(mu + jax.random.normal(k, mu.shape) * jp.exp(log_std), -1, 1)
            lp = log_prob(mu, log_std, action)
            ctrl = action.reshape(action.shape[0], -1)               # (E,64)
            data = vstep(data, ctrl)
            rew, done = vrew(data)                                   # (E,4),(E,)
            key, k = jax.random.split(key)
            rdata = vreset(jax.random.split(k, action.shape[0]))
            data = jax.tree.map(
                lambda a, b: jp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), data, rdata)
            nobs = vobs(data)
            return (data, nobs, key), (obs, action, lp, v, rew, done)
        carry, traj = jax.lax.scan(stepfn, (data, obs, key), None, length=T)
        data, obs, key = carry
        return traj, nforward(params, obs, mean, std)[1], data, obs, key

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
        obs, action, old_lp, val, rew, done = traj          # done (T,E); broadcast to players
        done_p = jp.broadcast_to(done[..., None], rew.shape)
        adv, vtgt = gae(rew, val, done_p, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape((-1,) + x.shape[3:])
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
        return params, opt_state, key, jp.mean(rew), obs.reshape(-1, obs.shape[-1])
    return rollout, update, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)   # 4 players each; >~384 OOMs 16GB VRAM
    ap.add_argument("--rollout", type=int, default=40)
    ap.add_argument("--updates", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--render-every", type=int, default=500)
    ap.add_argument("--init", type=str, default=str(ROOT / "checkpoints" / "walk_fwd.pkl"),
                    help="locomotion checkpoint to warm-start from")
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "soccer.pkl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--play", type=str, default="")
    args = ap.parse_args()

    env = SoccerEnv()
    if args.play:
        _play(env, load_ckpt(Path(args.play)), ROOT / "soccer.gif")
        return

    E, T = args.envs, args.rollout
    run = ROOT / "runs" / datetime.datetime.now().strftime("soccer_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a", buffering=1)
    def log(m):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(line); logf.write(line + "\n")

    key = jax.random.PRNGKey(0)
    key, s = jax.random.split(key)
    params, mean, m2, count, warm = transfer_from_walk(args.init, env.obs_dim, 16, s)
    log(f"JAX {jax.default_backend()} | {E} envs x {NP} players | obs={env.obs_dim} | warm-start={warm}")
    rollout, update, opt = build(env, E, T, args.epochs)
    opt_state = opt.init(params); start = 0
    if args.resume and Path(args.ckpt).exists():
        st = load_ckpt(Path(args.ckpt)); params, opt_state = st["params"], st["opt_state"]
        mean, m2, count, start = st["mean"], st["m2"], st["count"], st["update"]
        log(f"resumed @ {start}")

    key, s = jax.random.split(key)
    data = jax.vmap(env.reset)(jax.random.split(s, E))
    obs = jax.vmap(env.observe)(data)

    def snap(u): save_ckpt(Path(args.ckpt), dict(params=params, opt_state=opt_state,
                                                 mean=mean, m2=m2, count=count, update=u))
    stop = {"v": False}; signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    t0 = time.time(); u = start
    for u in range(start, args.updates):
        std = jp.sqrt(m2 / count) + 1e-5
        traj, last_v, data, obs, key = rollout(params, mean, std, data, obs, key)
        params, opt_state, key, mr, fo = update(params, opt_state, mean, std, traj, last_v, key)
        mean, m2, count = norm_update(mean, m2, count, fo)
        if u % 10 == 0:
            log(f"update {u:6d} | reward {float(mr):8.3f} | {E*NP*T*(u-start+1)/(time.time()-t0)/1e6:5.2f}M/s")
        if u > start and u % args.save_every == 0:
            snap(u); log(f"  ckpt @ {u}")
        if args.render_every and u > start and u % args.render_every == 0:
            try: _render(env, params, mean, m2, count, run / f"frame_{u:06d}.png"); log(f"  render @ {u}")
            except Exception as e: log(f"  render failed: {e}")
        if (args.hours and time.time() - t0 >= args.hours * 3600) or stop["v"]:
            log("stopping"); break
    snap(u + 1); log(f"done. ckpt {args.ckpt}"); logf.close()


# ---------------------------------------------------------------- render

def _rollout_match(env, params, mean, std, steps):
    dx = env.reset(jax.random.PRNGKey(1)); step = jax.jit(env.step)
    qframes = []
    for _ in range(steps):
        obs = env.observe(dx)
        mu, _, _ = nforward(params, obs, mean, std)
        dx = step(dx, jp.clip(mu, -1, 1).reshape(-1))
        _, done = env.reward_done(dx)
        qframes.append(np.array(dx.qpos))
        if bool(done): dx = env.reset(jax.random.PRNGKey(int(dx.qpos[0] * 1e3) % 100000))
    return qframes


def _cam():
    import mujoco
    c = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c)
    c.distance = 16; c.elevation = -40; c.azimuth = 90; c.lookat[:] = [0, 0, 0.4]
    return c


def _render(env, params, mean, m2, count, out_png):
    import mujoco, imageio
    os.environ.setdefault("MUJOCO_GL", "egl")
    std = jp.sqrt(m2 / count) + 1e-5
    qp = _rollout_match(env, params, mean, std, 150)[-1]
    d = mujoco.MjData(env.mj); d.qpos[:] = qp; mujoco.mj_forward(env.mj, d)
    with mujoco.Renderer(env.mj, 720, 1280) as r:
        r.update_scene(d, _cam()); imageio.imwrite(str(out_png), r.render())


def _play(env, st, out_gif, steps=400):
    import mujoco, imageio
    os.environ.setdefault("MUJOCO_GL", "egl")
    std = jp.sqrt(st["m2"] / st["count"]) + 1e-5
    qframes = _rollout_match(env, st["params"], st["mean"], std, steps)
    d = mujoco.MjData(env.mj); cam = _cam(); frames = []
    with mujoco.Renderer(env.mj, 720, 1280) as r:
        for qp in qframes:
            d.qpos[:] = qp; mujoco.mj_forward(env.mj, d); r.update_scene(d, cam); frames.append(r.render())
    imageio.mimsave(str(out_gif), frames, fps=30)
    print(f"wrote {out_gif}")


if __name__ == "__main__":
    main()
