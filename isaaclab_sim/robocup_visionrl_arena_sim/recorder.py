from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._bootstrap import args_cli
if args_cli.record_video:
    import cv2
    import numpy as np
    import torch

from ._bootstrap import (
    BASE_ARMOR,
    BLUE_ROBOT_PATH,
    Camera,
    CameraCfg,
    MATCH_CONTROLLERS,
    MATCH_STATE,
    RECORDING_POV_CAMERA_POSE,
    YELLOW_ROBOT_PATH,
    args_cli,
    sim_utils
)
from .rules import opponent_team

class MatchVideoRecorder:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.view = args_cli.record_view
        self.fps = max(1, int(args_cli.record_fps))
        self.width = max(320, int(args_cli.record_width))
        self.height = max(240, int(args_cli.record_height))
        self.panel_width = min(420, max(340, int(self.width * 0.26)))
        self.scene_width = max(480, self.width - self.panel_width)
        self.scene_height = self.height
        self.next_frame_time = 0.0
        self.frame_count = 0
        camera_prim_path = "/World/RecordingCamera"
        camera_offset = CameraCfg.OffsetCfg()
        camera_spawn = sim_utils.PinholeCameraCfg(
            focal_length=31.0,
            focus_distance=3.8,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 100.0),
        )
        if self.view in ("yellow_pov", "blue_pov"):
            robot_path = YELLOW_ROBOT_PATH if self.view == "yellow_pov" else BLUE_ROBOT_PATH
            camera_prim_path = f"{robot_path}/PovRecordingCamera"
            camera_spawn = sim_utils.PinholeCameraCfg(
                focal_length=2.6,
                focus_distance=2.0,
                horizontal_aperture=7.2,
                clipping_range=(0.02, 6.0),
            )
            camera_offset = CameraCfg.OffsetCfg(
                pos=RECORDING_POV_CAMERA_POSE,
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            )
        self.camera = Camera(
            CameraCfg(
                prim_path=camera_prim_path,
                update_period=0.0,
                height=self.scene_height,
                width=self.scene_width,
                data_types=["rgb"],
                spawn=camera_spawn,
                offset=camera_offset,
            )
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.output_path}")

    def initialize_view(self):
        if self.view not in ("overview", "top"):
            return
        if self.view == "top":
            eye = torch.tensor([[0.0, 0.0, 4.85]], dtype=torch.float32, device=self.camera.device)
            target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.camera.device)
        else:
            eye = torch.tensor([[2.15, -2.55, 2.28]], dtype=torch.float32, device=self.camera.device)
            target = torch.tensor([[0.0, 0.0, 0.12]], dtype=torch.float32, device=self.camera.device)
        self.camera.set_world_poses_from_view(eye, target)

    def capture(self, sim_dt: float, match_time: float):
        if match_time + 1e-6 < self.next_frame_time:
            return
        self.camera.update(dt=sim_dt)
        rgb = self.camera.data.output.get("rgb")
        if rgb is None:
            return
        frame = rgb[0].detach().cpu().numpy() if hasattr(rgb, "detach") else rgb[0]
        if frame.dtype != np.uint8:
            scale = 255.0 if frame.max() <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        scene_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if scene_bgr.shape[0] != self.scene_height or scene_bgr.shape[1] != self.scene_width:
            scene_bgr = cv2.resize(scene_bgr, (self.scene_width, self.scene_height), interpolation=cv2.INTER_AREA)
        output = self._compose_frame(scene_bgr, match_time)
        self.writer.write(output)
        self.frame_count += 1
        self.next_frame_time += 1.0 / float(self.fps)

    def _compose_frame(self, scene_bgr, match_time: float):
        frame = np.full((self.height, self.width, 3), 246, dtype=np.uint8)
        frame[:, : self.scene_width] = scene_bgr
        cv2.line(frame, (self.scene_width, 0), (self.scene_width, self.height), (190, 190, 190), 2)
        self._draw_side_panel(frame, match_time)
        return frame

    def _put(self, frame, text: str, x: int, y: int, scale: float, color=(35, 35, 35), thickness: int = 2):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _put_wrapped(self, frame, text: str, x: int, y: int, max_chars: int, line_gap: int, scale: float):
        words = str(text).split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if len(candidate) > max_chars:
                self._put(frame, line, x, y, scale, (40, 40, 40), 2)
                y += line_gap
                line = word
            else:
                line = candidate
        if line:
            self._put(frame, line, x, y, scale, (40, 40, 40), 2)
        return y

    def _view_title(self) -> str:
        if self.view == "yellow_pov":
            return "Yellow Robot POV"
        if self.view == "blue_pov":
            return "Blue Robot POV"
        if self.view == "top":
            return "Top View"
        return "Complete Arena View"

    def _opponent_summary(self, team: str) -> str:
        controller = MATCH_CONTROLLERS.get(team)
        if controller is None or not controller.opponent_estimate["available"]:
            return f"{team[0].upper()} track pending"
        estimate = controller.opponent_estimate
        opponent = opponent_team(team)
        visible = "vis" if estimate["visible"] else "occ"
        return (
            f"{team[0].upper()}->{opponent[0].upper()} "
            f"d {float(estimate['distance']):.2f}m "
            f"b {math.degrees(float(estimate['relative_bearing'])):+.0f}deg "
            f"{visible} th {float(estimate['threat_to_own_base']):.2f}"
        )

    def _draw_side_panel(self, frame, match_time: float):
        x0 = self.scene_width
        pad = 24
        panel_x = x0 + pad
        right = self.width - pad
        cv2.rectangle(frame, (x0, 0), (self.width, self.height), (247, 248, 250), -1)
        cv2.putText(
            frame,
            "RoboCup VisionRL",
            (panel_x, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        self._put(frame, self._view_title(), panel_x, 88, 0.55, (70, 70, 70), 2)

        cv2.rectangle(frame, (panel_x, 124), (right, 248), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 124), (right, 248), (216, 220, 226), 1)
        self._put(frame, "Score", panel_x + 16, 158, 0.58, (50, 50, 50), 2)
        self._put(frame, f"Yellow  {MATCH_STATE['score_yellow']}", panel_x + 18, 196, 0.68, (30, 130, 150), 2)
        self._put(frame, f"Blue    {MATCH_STATE['score_blue']}", panel_x + 18, 232, 0.68, (170, 80, 25), 2)

        cv2.rectangle(frame, (panel_x, 278), (right, 438), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 278), (right, 438), (216, 220, 226), 1)
        self._put(frame, "Match State", panel_x + 16, 312, 0.58, (50, 50, 50), 2)
        self._put(frame, f"time   {match_time:05.1f}s", panel_x + 18, 348, 0.56, (40, 40, 40), 2)
        self._put(frame, f"armor  Y:{len(BASE_ARMOR['yellow'])}  B:{len(BASE_ARMOR['blue'])}", panel_x + 18, 382, 0.56, (40, 40, 40), 2)
        self._put(frame, self._opponent_summary("yellow"), panel_x + 18, 410, 0.42, (40, 40, 40), 1)
        self._put(frame, self._opponent_summary("blue"), panel_x + 18, 432, 0.42, (40, 40, 40), 1)

        cv2.rectangle(frame, (panel_x, 466), (right, 590), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 466), (right, 590), (216, 220, 226), 1)
        self._put(frame, "Latest Event", panel_x + 16, 500, 0.58, (50, 50, 50), 2)
        self._put_wrapped(frame, str(MATCH_STATE["last_event"]), panel_x + 18, 538, 30, 30, 0.48)

        cv2.rectangle(frame, (panel_x, 620), (right, 780), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, 620), (right, 780), (216, 220, 226), 1)
        self._put(frame, "Rule Gate", panel_x + 16, 654, 0.58, (50, 50, 50), 2)
        self._put(frame, "opponent targets only", panel_x + 18, 692, 0.50, (24, 120, 78), 2)
        self._put(frame, "normal hit -> +5", panel_x + 18, 726, 0.50, (40, 40, 40), 2)
        self._put(frame, "base hit -> win", panel_x + 18, 760, 0.50, (40, 40, 40), 2)

        winner = MATCH_STATE["winner"]
        if winner is not None:
            cv2.rectangle(frame, (panel_x, self.height - 100), (right, self.height - 32), (42, 170, 74), -1)
            self._put(frame, f"WINNER: {str(winner).upper()}", panel_x + 18, self.height - 58, 0.64, (255, 255, 255), 2)

    def close(self):
        self.writer.release()
        print(f"[VIDEO]: Wrote {self.frame_count} frames to {self.output_path}")
