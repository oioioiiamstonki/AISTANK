"""
arena.py — build one MuJoCo world holding N humanoids that DON'T collide with
each other (only with the floor), each a distinct color.

Collision filtering (MuJoCo contype/conaffinity — two geoms collide iff
(contype_a & conaffinity_b) | (contype_b & conaffinity_a) != 0):
    floor:          contype=1  conaffinity=2
    every humanoid: contype=2  conaffinity=1
  → humanoid·floor collides, humanoid·humanoid never does.

The N humanoids are laid out on a grid and each training agent controls one of
them, so you can watch a whole crowd learn at once.
"""
import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "assets" / "humanoid16.xml"


def _agent_color(i, n):
    h = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def _suffix_names(elem, suffix):
    """Append suffix to every name/joint reference in a subtree."""
    for e in elem.iter():
        if "name" in e.attrib:
            e.set("name", e.attrib["name"] + suffix)
        if e.tag == "joint" and "name" in e.attrib:
            pass  # already handled above
        if "joint" in e.attrib:           # actuators reference joints
            e.set("joint", e.attrib["joint"] + suffix)


def build_arena_xml(n_agents=16, spacing=2.0):
    tree = ET.parse(SRC)
    root = tree.getroot()

    # Drop the CPU-arena memory cap; let MuJoCo size the N-humanoid model.
    for sz in root.findall("size"):
        root.remove(sz)

    # Larger offscreen framebuffer for crowd renders.
    if root.find("visual") is None:
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})

    worldbody = root.find("worldbody")
    actuator = root.find("actuator")

    # Detach the template torso body and the motor list; keep the floor.
    torso = None
    for b in worldbody.findall("body"):
        if b.attrib.get("name") == "torso":
            torso = b
            break
    worldbody.remove(torso)
    motors = list(actuator)
    for m in motors:
        actuator.remove(m)

    # Floor collision class.
    floor = worldbody.find("geom")
    floor.set("contype", "1")
    floor.set("conaffinity", "2")

    cols = int(n_agents ** 0.5 + 0.999) or 1
    for i in range(n_agents):
        body = ET.fromstring(ET.tostring(torso))     # deep copy
        _suffix_names(body, f"_{i}")
        gx = (i % cols - cols / 2) * spacing
        gy = (i // cols - cols / 2) * spacing
        body.set("pos", f"{gx:.3f} {gy:.3f} 1.05")
        color = _agent_color(i, n_agents)
        for g in body.iter("geom"):
            g.set("rgba", color)
            g.set("contype", "2")
            g.set("conaffinity", "1")
        worldbody.append(body)
        for m in motors:
            nm = ET.fromstring(ET.tostring(m))
            _suffix_names(nm, f"_{i}")
            actuator.append(nm)

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    # Smoke test: build, load, render one frame of the colored crowd.
    import os, sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from PIL import Image

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    xml = build_arena_xml(n)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"arena: {n} humanoids | nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0 * n ** 0.5
    cam.elevation = -20
    cam.azimuth = 90
    cam.lookat[:] = [0, 0, 0.6]
    with mujoco.Renderer(m, 720, 1280) as r:
        r.update_scene(d, cam)
        img = r.render()
    Image.fromarray(img).save("mjx/arena_preview.png")
    print("wrote mjx/arena_preview.png")
