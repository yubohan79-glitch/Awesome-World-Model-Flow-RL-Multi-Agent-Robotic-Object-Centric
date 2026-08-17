from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher
from pxr import Usd


PUSHABLE_NAMES = ("RandomObstacleNorthEast", "RandomObstacleSouthWest")
RIGID_ATTRS = (
    "physics:rigidBodyEnabled",
    "physics:kinematicEnabled",
    "physics:disableGravity",
    "physics:mass",
    "physxRigidBody:linearDamping",
    "physxRigidBody:angularDamping",
    "physxRigidBody:solverPositionIterationCount",
    "physxRigidBody:solverVelocityIterationCount",
)
MESH_ATTRS = ("physics:collisionEnabled", "physxCollision:contactOffset", "physxCollision:restOffset")
MATERIAL_ATTRS = (
    "physics:staticFriction",
    "physics:dynamicFriction",
    "physics:restitution",
    "physxMaterial:frictionCombineMode",
    "physxMaterial:restitutionCombineMode",
)


def authored_value(prim: Usd.Prim, attr_name: str):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasAuthoredValueOpinion():
        value = attr.Get()
        return str(value) if value.__class__.__name__ == "TfToken" else value
    return None


def find_prim_by_name(stage, name: str):
    return next((prim for prim in stage.Traverse() if prim.GetName() == name), None)


def inspect_pushables(usd_path: Path) -> dict[str, object]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")
    result: dict[str, object] = {}
    for name in PUSHABLE_NAMES:
        prim = find_prim_by_name(stage, name)
        if prim is None:
            result[name] = {"present": False}
            continue
        mesh = stage.GetPrimAtPath(str(prim.GetPath()) + "/geometry/mesh")
        material = stage.GetPrimAtPath(str(prim.GetPath()) + "/geometry/material")
        result[name] = {
            "present": True,
            "path": str(prim.GetPath()),
            "rigid": {attr: authored_value(prim, attr) for attr in RIGID_ATTRS},
            "mesh": {attr: authored_value(mesh, attr) for attr in MESH_ATTRS},
            "material": {attr: authored_value(material, attr) for attr in MATERIAL_ATTRS},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect IsaacLab pushable-box USD physics attributes.")
    parser.add_argument("usd", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    result = inspect_pushables(args.usd)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    app_launcher.app.close()


if __name__ == "__main__":
    main()
