"""
soccer_scene.py — build a 2v2 soccer pitch MJCF: ground, two goals, four
team-colored humanoids, and a ball.

Team A (blue shades) defends the -x goal and attacks +x; team B (red shades)
the reverse. Everything collides (players<->players, players<->ball,
ball<->ground/goals); the ball is a free body with a springy (slightly bouncy)
contact. Aerodynamic curve (Magnus) + drag are applied in soccer_env.py, not
here — MuJoCo has no built-in lift on a spinning ball.

Field constants are exported so the env can compute goals / out-of-bounds.
"""
import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path

from arena import _suffix_names

SRC = Path(__file__).resolve().parent.parent / "assets" / "humanoid16.xml"

# Pitch geometry (metres). Small, for faster learning.
FIELD_LEN = 12.0      # along x (goal to goal)
FIELD_WID = 8.0       # along y
GOAL_WID = 2.6        # goal mouth width (y)
GOAL_HEIGHT = 1.3
BALL_R = 0.11
BALL_MASS = 0.43

# Player spawn positions (x, y) and facing (+1 attacks +x, -1 attacks -x).
SPAWNS = [
    (-2.5, -1.6, +1),   # A0
    (-2.5,  1.6, +1),   # A1
    ( 2.5, -1.6, -1),   # B0
    ( 2.5,  1.6, -1),   # B1
]
TEAM = [0, 0, 1, 1]
TEAM_HUES = [0.60, 0.02]          # blue-ish, red-ish
NUM_PLAYERS = 4


def _player_color(i):
    hue = TEAM_HUES[TEAM[i]]
    # two distinct shades per team
    v = 0.95 if (i % 2 == 0) else 0.65
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, v)
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def _yaw_quat(face):
    # Rotate 0 (face +x) or 180deg about z (face -x).
    return "1 0 0 0" if face > 0 else "0 0 0 1"


def build_soccer_xml():
    tree = ET.parse(SRC)
    root = tree.getroot()
    for sz in root.findall("size"):
        root.remove(sz)
    if root.find("visual") is None:
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})

    worldbody = root.find("worldbody")
    actuator = root.find("actuator")

    torso = next(b for b in worldbody.findall("body") if b.attrib.get("name") == "torso")
    worldbody.remove(torso)
    motors = list(actuator)
    for m in motors:
        actuator.remove(m)

    floor = worldbody.find("geom")
    floor.set("size", f"{FIELD_LEN} {FIELD_WID} 0.1")
    floor.set("rgba", "0.30 0.55 0.30 1")
    floor.set("contype", "1"); floor.set("conaffinity", "1")

    # --- goals: two posts + crossbar at each end (static) ---
    def goal(side):
        gx = side * FIELD_LEN / 2.0
        post = 0.06
        for sy in (-1, 1):
            b = ET.SubElement(worldbody, "body",
                              {"name": f"goalpost_{side}_{sy}",
                               "pos": f"{gx} {sy*GOAL_WID/2:.3f} {GOAL_HEIGHT/2:.3f}"})
            ET.SubElement(b, "geom", {"type": "box",
                                      "size": f"{post} {post} {GOAL_HEIGHT/2:.3f}",
                                      "rgba": "0.95 0.95 0.95 1",
                                      "contype": "1", "conaffinity": "1"})
        bar = ET.SubElement(worldbody, "body",
                            {"name": f"goalbar_{side}",
                             "pos": f"{gx} 0 {GOAL_HEIGHT:.3f}"})
        ET.SubElement(bar, "geom", {"type": "box",
                                    "size": f"{post} {GOAL_WID/2:.3f} {post}",
                                    "rgba": "0.95 0.95 0.95 1",
                                    "contype": "1", "conaffinity": "1"})
    goal(-1); goal(+1)

    # --- players ---
    for i in range(NUM_PLAYERS):
        body = ET.fromstring(ET.tostring(torso))
        _suffix_names(body, f"_p{i}")
        x, y, face = SPAWNS[i]
        body.set("pos", f"{x} {y} 1.05")
        body.set("quat", _yaw_quat(face))
        color = _player_color(i)
        for g in body.iter("geom"):
            g.set("rgba", color)
            g.set("contype", "1"); g.set("conaffinity", "1")
        worldbody.append(body)
        for m in motors:
            nm = ET.fromstring(ET.tostring(m))
            _suffix_names(nm, f"_p{i}")
            actuator.append(nm)

    # --- ball (free body, slightly bouncy contact) ---
    ball = ET.SubElement(worldbody, "body", {"name": "ball", "pos": f"0 0 {BALL_R}"})
    ET.SubElement(ball, "freejoint", {"name": "ball_free"})
    ET.SubElement(ball, "geom", {
        "name": "ball", "type": "sphere", "size": f"{BALL_R}",
        "mass": f"{BALL_MASS}", "rgba": "0.95 0.85 0.1 1",
        "contype": "1", "conaffinity": "1",
        "condim": "6", "friction": "0.7 0.05 0.02",
        "solref": "0.015 0.6",          # underdamped -> some bounce
    })

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image

    xml = build_soccer_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"soccer scene: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = 16; cam.elevation = -35; cam.azimuth = 90; cam.lookat[:] = [0, 0, 0.5]
    with mujoco.Renderer(m, 720, 1280) as r:
        r.update_scene(d, cam)
        Image.fromarray(r.render()).save("mjx/soccer_scene.png")
    print("wrote mjx/soccer_scene.png")
