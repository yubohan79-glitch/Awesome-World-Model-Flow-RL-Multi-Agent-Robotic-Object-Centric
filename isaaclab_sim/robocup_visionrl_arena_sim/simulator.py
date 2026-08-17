from __future__ import annotations

import math
import os
import sys
import time


from ._bootstrap import (
    BASE_ARMOR,
    BLUE_ROBOT_PATH,
    BLUE_START_XY,
    COLLISION_PRIMS,
    MATCH_CONTROLLERS,
    MATCH_DURATION_S,
    MATCH_STATE,
    YELLOW_ROBOT_PATH,
    YELLOW_START_XY,
    args_cli,
    sim_utils,
    simulation_app
)
from .laser import (
    export_stage,
    update_armor_removals,
    update_target_contacts,
    update_target_falls
)
from .recorder import MatchVideoRecorder
from .replay import sync_pushable_obstacles_from_stage
from .scene import (
    create_sensor_streams,
    design_arena,
    design_robot,
    update_robot_animation
)

def run_simulator(sim: sim_utils.SimulationContext, sensors: dict[str, object], recorder: MatchVideoRecorder | None):
    sim_dt = sim.get_physics_dt()
    start = time.perf_counter()
    count = 0
    last_print = -1.0

    try:
        while simulation_app.is_running():
            elapsed = time.perf_counter() - start
            MATCH_STATE["current_time"] = elapsed
            robot_poses = update_robot_animation(elapsed)
            if not args_cli.replay_trace:
                update_target_contacts(robot_poses)
            update_armor_removals(elapsed)
            update_target_falls(elapsed)

            if MATCH_STATE["winner"] is None and elapsed >= MATCH_DURATION_S:
                yellow_score = int(MATCH_STATE["score_yellow"])
                blue_score = int(MATCH_STATE["score_blue"])
                if yellow_score > blue_score:
                    MATCH_STATE["winner"] = "yellow"
                elif blue_score > yellow_score:
                    MATCH_STATE["winner"] = "blue"
                else:
                    MATCH_STATE["winner"] = "draw"
                MATCH_STATE["last_event"] = f"time limit reached; winner={MATCH_STATE['winner']}"
                print(f"[RULE]: 3 minute time limit reached. winner={MATCH_STATE['winner']}.")

            sim.step()
            sync_pushable_obstacles_from_stage()

            if "camera" in sensors:
                sensors["camera"].update(dt=sim_dt)
            if "lidar" in sensors:
                sensors["lidar"].update(dt=sim_dt, force_recompute=True)
            if "imu" in sensors:
                sensors["imu"].update(dt=sim_dt)
            if recorder is not None:
                recorder.capture(sim_dt, elapsed)

            if elapsed - last_print > 4.0:
                last_print = elapsed
                print("[INFO]: RoboCup VisionRL two-robot scene running")
                print(
                    "[SCORE]: "
                    f"blue_armor={len(BASE_ARMOR['blue'])} yellow_armor={len(BASE_ARMOR['yellow'])} "
                    f"yellow_score={MATCH_STATE['score_yellow']} blue_score={MATCH_STATE['score_blue']} "
                    f"winner={MATCH_STATE['winner']}"
                )
                if MATCH_CONTROLLERS:
                    print(
                        "[LOCALIZATION]: "
                        f"yellow_conf={MATCH_CONTROLLERS['yellow'].localization_confidence:.2f} "
                        f"blue_conf={MATCH_CONTROLLERS['blue'].localization_confidence:.2f}"
                    )
                    yellow_track = MATCH_CONTROLLERS["yellow"].opponent_estimate
                    blue_track = MATCH_CONTROLLERS["blue"].opponent_estimate
                    print(
                        "[OPPONENT_TRACK]: "
                        f"yellow_to_blue d={float(yellow_track['distance']):.2f}m "
                        f"bearing={math.degrees(float(yellow_track['relative_bearing'])):+.1f}deg "
                        f"visible={bool(yellow_track['visible'])} threat={float(yellow_track['threat_to_own_base']):.2f}; "
                        f"blue_to_yellow d={float(blue_track['distance']):.2f}m "
                        f"bearing={math.degrees(float(blue_track['relative_bearing'])):+.1f}deg "
                        f"visible={bool(blue_track['visible'])} threat={float(blue_track['threat_to_own_base']):.2f}"
                    )
                if "camera" in sensors and sensors["camera"].data.output:
                    rgb = sensors["camera"].data.output.get("rgb")
                    depth = sensors["camera"].data.output.get("distance_to_image_plane")
                    print(f"[INFO]: camera rgb={None if rgb is None else tuple(rgb.shape)} depth={None if depth is None else tuple(depth.shape)}")
                if "lidar" in sensors:
                    print(f"[INFO]: lidar rays={sensors['lidar'].num_rays} targets={len(COLLISION_PRIMS)}")
                if "imu" in sensors:
                    imu_data = getattr(sensors["imu"], "data", None)
                    print(f"[INFO]: imu stream={'ready' if imu_data is not None else 'pending'}")

            count += 1
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
    finally:
        if recorder is not None:
            recorder.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.15, -3.65, 2.45], target=[0.0, 0.0, 0.20])

    design_arena()
    design_robot(
        YELLOW_ROBOT_PATH,
        YELLOW_START_XY,
        math.pi * 0.5,
        team_color=(0.95, 0.86, 0.08),
        accent_color=(0.64, 0.48, 0.10),
        beam_color=(1.0, 0.08, 0.02),
    )
    design_robot(
        BLUE_ROBOT_PATH,
        BLUE_START_XY,
        -math.pi * 0.5,
        team_color=(0.12, 0.36, 0.90),
        accent_color=(0.08, 0.20, 0.56),
        beam_color=(0.15, 0.42, 1.0),
    )
    export_stage()
    sensors = create_sensor_streams()
    recorder = MatchVideoRecorder(args_cli.record_video) if args_cli.record_video else None

    sim.reset()
    if recorder is not None:
        recorder.initialize_view()
    print("[INFO]: Setup complete. Close the Isaac Sim window to stop the scene.")
    print("[INFO]: Field: 3m x 3m, regulation-aligned bases/start zones, 0.5m walls, 0.3m obstacles.")
    print("[INFO]: Robots: yellow and blue 0.34m L x 0.24m W x 0.245m H, camera, 2D lidar, fixed laser module.")
    run_simulator(sim, sensors, recorder)


if __name__ == "__main__":
    main()
    if args_cli.headless and args_cli.duration > 0.0:
        print("[INFO]: Headless timed run complete; exiting process without Kit shutdown cleanup.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
