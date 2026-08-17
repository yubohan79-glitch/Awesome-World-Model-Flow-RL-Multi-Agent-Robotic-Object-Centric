from __future__ import annotations

import math


from ._bootstrap import (
    Gf,
    UsdGeom,
    get_current_stage
)

def quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return USD quaternion order (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def rotate_local(offset: tuple[float, float, float], roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    """Apply Rz(yaw) * Ry(pitch) * Rx(roll) to a local vector."""
    x, y, z = offset

    cr = math.cos(roll)
    sr = math.sin(roll)
    y, z = cr * y - sr * z, sr * y + cr * z

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x, z = cp * x + sp * z, -sp * x + cp * z

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x, y = cy * x - sy * y, sy * x + cy * y
    return (x, y, z)


def quat_rotate(
    quat: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vector
    # q * v * q^-1, expanded to avoid extra dependencies.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def local_to_world(
    origin: tuple[float, float, float],
    offset: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float]:
    dx, dy, dz = rotate_local(offset, roll, pitch, yaw)
    return (origin[0] + dx, origin[1] + dy, origin[2] + dz)


def create_xform(
    path: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    if translation is not None or orientation is not None:
        set_xform(path, translation or (0.0, 0.0, 0.0), orientation or (1.0, 0.0, 0.0, 0.0))
    return prim


def set_xform(path: str, translation: tuple[float, float, float], orientation: tuple[float, float, float, float]):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {path}")

    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    translate_op = ops.get("xformOp:translate")
    orient_op = ops.get("xformOp:orient")
    scale_op = ops.get("xformOp:scale")
    if translate_op is None:
        translate_op = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble)
    if scale_op is None:
        scale_op = xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
    translate_op.Set(Gf.Vec3d(*translation))
    orient_op.Set(Gf.Quatd(orientation[0], Gf.Vec3d(orientation[1], orientation[2], orientation[3])))
    xform.SetXformOpOrder([translate_op, orient_op, scale_op])


def get_xform(path: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {path}")

    translation = (0.0, 0.0, 0.0)
    orientation = (1.0, 0.0, 0.0, 0.0)
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            value = op.Get()
            translation = (float(value[0]), float(value[1]), float(value[2]))
        elif op.GetOpName() == "xformOp:orient":
            value = op.Get()
            imag = value.GetImaginary()
            orientation = (float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    return translation, orientation


def set_visibility(path: str, visible: bool):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()

