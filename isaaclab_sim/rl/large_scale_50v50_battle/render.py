from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    BattleConfig,
    DATA_DIR,
    MEDIA_DIR,
    config_from_args
)
from .sim import (
    LargeScaleBattle50v50
)
from .train import (
    load_checkpoint
)

def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def world_to_px(pos: np.ndarray, width: int, height: int, cfg: BattleConfig) -> np.ndarray:
    scale = min((width - 120) / cfg.width_m, (height - 120) / cfg.height_m)
    ox = (width - cfg.width_m * scale) / 2.0
    oy = (height - cfg.height_m * scale) / 2.0
    out = np.empty_like(pos, dtype=np.float64)
    out[:, 0] = ox + pos[:, 0] * scale
    out[:, 1] = oy + (cfg.height_m - pos[:, 1]) * scale
    return out


def render_frame(env: LargeScaleBattle50v50, item: dict[str, Any], width: int = 1920, height: int = 1080) -> Image.Image:
    cfg = env.cfg
    img = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(img)
    title_font = _font(34)
    label_font = _font(22)
    small_font = _font(18)
    scale = min((width - 120) / cfg.width_m, (height - 120) / cfg.height_m)
    ox = (width - cfg.width_m * scale) / 2.0
    oy = (height - cfg.height_m * scale) / 2.0

    def rect_world(rect, fill, outline="#334155"):
        xmin, ymin, xmax, ymax = rect
        x1 = ox + xmin * scale
        x2 = ox + xmax * scale
        y1 = oy + (cfg.height_m - ymax) * scale
        y2 = oy + (cfg.height_m - ymin) * scale
        draw.rounded_rectangle([x1, y1, x2, y2], radius=6, fill=fill, outline=outline, width=2)

    draw.rectangle([ox, oy, ox + cfg.width_m * scale, oy + cfg.height_m * scale], fill="#ffffff", outline="#0f172a", width=3)
    for i in range(1, 4):
        x = ox + cfg.width_m * scale * i / 4
        draw.line([x, oy, x, oy + cfg.height_m * scale], fill="#e2e8f0", width=1)
    for i in range(1, 4):
        y = oy + cfg.height_m * scale * i / 4
        draw.line([ox, y, ox + cfg.width_m * scale, y], fill="#e2e8f0", width=1)
    for rect in env.obstacles:
        rect_world(rect, "#cbd5e1")
    for idx, zone in enumerate(env.zones):
        state = item["zone_state"][idx]
        color = "#facc15" if state > 0.25 else "#3b82f6" if state < -0.25 else "#e2e8f0"
        center = world_to_px(zone[None, :], width, height, cfg)[0]
        r = cfg.capture_radius_m * scale
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], outline=color, width=5)
        draw.text((center[0] - 10, center[1] - 12), str(idx + 1), font=label_font, fill="#0f172a")

    for base, color, hp in [(env.yellow_base, "#eab308", item["yellow_base_hp"]), (env.blue_base, "#2563eb", item["blue_base_hp"])]:
        p = world_to_px(base[None, :], width, height, cfg)[0]
        draw.rounded_rectangle([p[0] - 24, p[1] - 34, p[0] + 24, p[1] + 34], radius=8, fill=color, outline="#0f172a", width=2)
        draw.rectangle([p[0] - 35, p[1] + 42, p[0] + 35, p[1] + 50], fill="#e5e7eb")
        draw.rectangle([p[0] - 35, p[1] + 42, p[0] - 35 + 70 * max(0.0, hp / cfg.base_hp), p[1] + 50], fill="#22c55e")

    for key, color, edge in [("yellow", "#facc15", "#854d0e"), ("blue", "#60a5fa", "#1e3a8a")]:
        pos = world_to_px(item[f"{key}_pos"], width, height, cfg)
        alive = item[f"{key}_alive"]
        for p, ok in zip(pos, alive):
            if ok:
                draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=color, outline=edge)
            else:
                draw.line([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill="#94a3b8", width=2)
                draw.line([p[0] - 4, p[1] + 4, p[0] + 4, p[1] - 4], fill="#94a3b8", width=2)

    y_alive = int(np.count_nonzero(item["yellow_alive"]))
    b_alive = int(np.count_nonzero(item["blue_alive"]))
    draw.text((70, 30), "Large-Scale 50v50 Multi-Agent Battle Replay", font=title_font, fill="#0f172a")
    draw.text(
        (70, 75),
        f"t={item['step'] * cfg.dt_s:.1f}s   yellow alive={y_alive}/{cfg.agents_per_team}   blue alive={b_alive}/{cfg.agents_per_team}",
        font=label_font,
        fill="#334155",
    )
    draw.text((width - 520, 35), f"base hp: Y {item['yellow_base_hp']:.1f} | B {item['blue_base_hp']:.1f}", font=label_font, fill="#334155")
    draw.text((width - 520, 75), "zones: yellow if gold, blue if blue, neutral if gray", font=small_font, fill="#64748b")
    return img


def render_video(args: argparse.Namespace) -> dict[str, str]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    ckpt = load_checkpoint(Path(args.checkpoint))
    theta = np.array(ckpt["theta"], dtype=np.float64)
    ep = env.run_episode(theta, theta, args.seed, collect_trace=True, trace_stride=args.trace_stride)
    trace = ep["trace"]
    np.savez_compressed(
        DATA_DIR / "isaaclab_replay_trace.npz",
        yellow_pos=np.stack([item["yellow_pos"] for item in trace], axis=0),
        blue_pos=np.stack([item["blue_pos"] for item in trace], axis=0),
        yellow_alive=np.stack([item["yellow_alive"] for item in trace], axis=0),
        blue_alive=np.stack([item["blue_alive"] for item in trace], axis=0),
        zone_state=np.stack([item["zone_state"] for item in trace], axis=0),
        yellow_base_hp=np.array([item["yellow_base_hp"] for item in trace], dtype=np.float32),
        blue_base_hp=np.array([item["blue_base_hp"] for item in trace], dtype=np.float32),
        yellow_base_open=np.array([item["yellow_base_open"] for item in trace], dtype=np.bool_),
        blue_base_open=np.array([item["blue_base_open"] for item in trace], dtype=np.bool_),
        dt=np.array([env.cfg.dt_s * args.trace_stride], dtype=np.float32),
        width_m=np.array([env.cfg.width_m], dtype=np.float32),
        height_m=np.array([env.cfg.height_m], dtype=np.float32),
    )
    mp4_path = MEDIA_DIR / "large_scale_50v50_replay.mp4"
    gif_path = MEDIA_DIR / "large_scale_50v50_replay.gif"
    fps = args.fps
    max_frames = max(1, int(args.seconds * fps))
    frame_indices = np.linspace(0, len(trace) - 1, max_frames).round().astype(int)
    gif_target = max(1, int(args.gif_seconds * args.gif_fps))
    gif_pick = set(np.linspace(0, max_frames - 1, gif_target).round().astype(int).tolist())
    gif_frames = []
    with imageio.get_writer(mp4_path, fps=fps, quality=8, macro_block_size=1) as writer:
        for out_idx, trace_idx in enumerate(frame_indices):
            frame = render_frame(env, trace[int(trace_idx)], args.width, args.height)
            writer.append_data(np.asarray(frame))
            if out_idx in gif_pick:
                gif_frames.append(frame.resize((960, 540), Image.Resampling.LANCZOS))
    imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps, loop=0)
    return {"mp4": str(mp4_path), "gif": str(gif_path)}
