"""
soccer_env.py — MJX 2v2 soccer environment with realistic ball aerodynamics.

The ball is a free rigid body. On top of MuJoCo's contact/friction/spin we add,
each physics substep, the two aerodynamic forces that make a real football move:

    Magnus (curve):  F_m = K_MAGNUS * (omega x v)     # spin bends the flight
    drag:            F_d = -K_DRAG * |v| * v           # quadratic air drag

applied via dx.xfrc_applied on the ball body. A ball struck with sidespin
therefore curves through the air, and a hard shot decelerates realistically.

Per-player observation = the 59-dim locomotion observation (identical to the
walk/imitation policy, so that skill transfers) + soccer context (ball,
teammate, opponents, target goal). Reward shapes: stay upright, get to the ball,
drive the ball toward the opponent goal, and score.
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from functools import partial
from mujoco import mjx

import soccer_scene as scene
from train_arena import humanoid_obs, QPOS_PER, QVEL_PER
from train_humanoid import quat_rotate, W_ALIVE, W_UPRIGHT, TERM_HEIGHT

NP = scene.NUM_PLAYERS              # 4
BALL_Q = NP * QPOS_PER             # 92: ball free-joint qpos start
BALL_V = NP * QVEL_PER             # 88: ball free-joint qvel start
HALF_LEN = scene.FIELD_LEN / 2.0
HALF_WID = scene.FIELD_WID / 2.0

K_MAGNUS = 0.0030                  # curve strength (tuned for a visible bend)
K_DRAG = 0.010
MAX_STEPS = 1000

# reward weights
W_TO_BALL = 0.05                   # closing distance to the ball
W_BALL_GOAL = 0.5                  # ball velocity toward opponent goal
W_GOAL = 50.0                      # scoring


def _quat_rotate(q, v):
    return quat_rotate(q, v)


class SoccerEnv:
    def __init__(self, n_substeps=5):
        self.mj = mujoco.MjModel.from_xml_string(scene.build_soccer_xml())
        self.mx = mjx.put_model(self.mj)
        self.n_substeps = n_substeps
        self.ball_bid = self.mj.body("ball").id
        self.team = jp.array(scene.TEAM)                  # (4,)
        # opponent goal x per player (+x for team 0, -x for team 1)
        self.attack_x = jp.where(self.team == 0, HALF_LEN, -HALF_LEN)
        self.spawn = jp.array([[s[0], s[1]] for s in scene.SPAWNS])
        self.face = jp.array([s[2] for s in scene.SPAWNS])
        self.qpos0 = jp.array(self.mj.qpos0)
        self.obs_dim = int(self.observe(self.reset(jax.random.PRNGKey(0)))[0].shape[-1])

    # ------------------------------------------------------------ dynamics

    def _aero(self, dx):
        v = dx.qvel[BALL_V:BALL_V + 3]
        w_local = dx.qvel[BALL_V + 3:BALL_V + 6]
        quat = dx.qpos[BALL_Q + 3:BALL_Q + 7]
        w = _quat_rotate(quat, w_local)
        speed = jp.sqrt(jp.sum(v * v) + 1e-9)
        f = K_MAGNUS * jp.cross(w, v) - K_DRAG * speed * v
        xfrc = dx.xfrc_applied.at[self.ball_bid, 0:3].set(f)
        return dx.replace(xfrc_applied=xfrc)

    @partial(jax.jit, static_argnums=0)
    def step(self, dx, ctrl):
        def body(d, _):
            d = self._aero(d.replace(ctrl=ctrl))
            return mjx.step(self.mx, d), None
        dx, _ = jax.lax.scan(body, dx, None, length=self.n_substeps)
        return dx

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        dx = mjx.make_data(self.mx)
        qpos = self.qpos0
        # small per-player joint jitter; ball at centre with a little randomness
        key, k = jax.random.split(key)
        jitter = 0.01 * jax.random.normal(k, (self.mx.nq,))
        jitter = jitter.at[BALL_Q:].set(0.0)
        qpos = qpos + jitter
        key, k = jax.random.split(key)
        bxy = 0.3 * jax.random.normal(k, (2,))
        qpos = qpos.at[BALL_Q:BALL_Q + 2].set(bxy)
        dx = dx.replace(qpos=qpos)
        return mjx.forward(self.mx, dx)

    # ------------------------------------------------------------ obs / reward

    def _player_slice(self, dx, p):
        qp = jax.lax.dynamic_slice_in_dim(dx.qpos, p * QPOS_PER, QPOS_PER)
        qv = jax.lax.dynamic_slice_in_dim(dx.qvel, p * QVEL_PER, QVEL_PER)
        return qp, qv

    def observe(self, dx):
        ball_pos = dx.qpos[BALL_Q:BALL_Q + 3]
        ball_vel = dx.qvel[BALL_V:BALL_V + 3]
        roots = jp.stack([dx.qpos[p * QPOS_PER:p * QPOS_PER + 3] for p in range(NP)])

        def one(p):
            qp, qv = self._player_slice(dx, p)
            loco = humanoid_obs(qp, qv, jp.zeros(16))          # 59-dim, matches walker
            root = roots[p]
            mates = [roots[q] - root for q in range(NP) if scene.TEAM[q] == scene.TEAM[p] and q != p]
            opps = [roots[q] - root for q in range(NP) if scene.TEAM[q] != scene.TEAM[p]]
            goal = jp.array([self.attack_x[p], 0.0, 0.0]) - root
            extra = jp.concatenate([ball_pos - root, ball_vel] + mates + opps + [goal])
            return jp.concatenate([loco, extra])

        return jp.stack([one(p) for p in range(NP)])           # (4, obs_dim)

    def reward_done(self, dx, prev_ball_x):
        ball = dx.qpos[BALL_Q:BALL_Q + 3]
        ball_v = dx.qvel[BALL_V:BALL_V + 3]
        rewards = []
        for p in range(NP):
            root = dx.qpos[p * QPOS_PER:p * QPOS_PER + 3]
            up = _quat_rotate(dx.qpos[p * QPOS_PER + 3:p * QPOS_PER + 7],
                              jp.array([0.0, 0.0, 1.0]))[2]
            d_ball = jp.sqrt(jp.sum((root - ball) ** 2) + 1e-6)
            toward = jp.sign(self.attack_x[p]) * ball_v[0]      # ball speed toward opp goal
            r = (W_ALIVE + W_UPRIGHT * up
                 - W_TO_BALL * d_ball
                 + W_BALL_GOAL * toward)
            rewards.append(r)
        rewards = jp.stack(rewards)

        # goal: ball past a goal line within the mouth
        in_mouth = (jp.abs(ball[1]) < scene.GOAL_WID / 2) & (ball[2] < scene.GOAL_HEIGHT)
        scored_px = (ball[0] > HALF_LEN) & in_mouth            # into +x goal (team0 scores)
        scored_nx = (ball[0] < -HALF_LEN) & in_mouth           # into -x goal (team1 scores)
        team_sign = jp.where(self.team == 0, 1.0, -1.0)
        rewards = rewards + W_GOAL * jp.where(scored_px, team_sign,
                                              jp.where(scored_nx, -team_sign, 0.0))

        out = (jp.abs(ball[0]) > HALF_LEN + 0.5) | (jp.abs(ball[1]) > HALF_WID + 0.5)
        fell = jp.array([dx.qpos[p * QPOS_PER + 2] < TERM_HEIGHT for p in range(NP)]).all()
        done = scored_px | scored_nx | out | fell
        return rewards, done


# ---------------------------------------------------------------- ball-curve test

def _ball_curve_test():
    """Kick the ball with sidespin and confirm the flight curves laterally."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = SoccerEnv()
    dx = env.reset(jax.random.PRNGKey(0))
    # launch ball: forward+up velocity, strong spin about the vertical axis
    qvel = dx.qvel
    qvel = qvel.at[BALL_V:BALL_V + 3].set(jp.array([8.0, 0.0, 3.0]))     # vx, vy, vz
    qvel = qvel.at[BALL_V + 3:BALL_V + 6].set(jp.array([0.0, 0.0, 60.0]))  # topspin-z
    dx = dx.replace(qvel=qvel)
    ctrl = jp.zeros(env.mj.nu)
    step = jax.jit(env.step)

    ys, xs = [], []
    for _ in range(60):
        dx = step(dx, ctrl)
        xs.append(float(dx.qpos[BALL_Q]))
        ys.append(float(dx.qpos[BALL_Q + 1]))
    lateral = ys[-1] - ys[0]
    print(f"ball travelled x={xs[-1]-xs[0]:.2f}m, lateral curve y={lateral:+.2f}m")
    print("CURVES" if abs(lateral) > 0.3 else "nearly straight (increase K_MAGNUS)")

    # also compare drag on vs off
    return lateral


if __name__ == "__main__":
    _ball_curve_test()
