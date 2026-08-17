from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from policies import FlowActor
from robocup_visionrl_selfplay_env import TACTICAL_ACTION_LABELS
from torch import nn
from train_world_model_sacflow_selfplay import MultiAgentFlowActors


class FlowActorOnly(nn.Module):
    """Deployment wrapper for the object-centric SAC Flow tactical actor."""

    def __init__(self, policy: MultiAgentFlowActors, export_team: str = "auto"):
        super().__init__()
        self.actor_mode = policy.actor_mode
        self.export_team = export_team
        if policy.actor_mode == "shared":
            self.shared_actor = policy.shared_actor
        else:
            self.yellow_actor = policy.yellow_actor
            self.blue_actor = policy.blue_actor

    def _dispatch(self, observation: torch.Tensor) -> FlowActor:
        if self.actor_mode == "shared":
            return self.shared_actor
        if self.export_team == "blue":
            return self.blue_actor
        return self.yellow_actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.actor_mode == "shared" or self.export_team in ("yellow", "blue"):
            return self._dispatch(observation).deterministic(observation)

        blue_rows = (observation[:, -1] < 0.0).reshape(-1, 1)
        yellow_action = self.yellow_actor.deterministic(observation)
        blue_action = self.blue_actor.deterministic(observation)
        return torch.where(blue_rows, blue_action, yellow_action)


def load_actor(checkpoint_path: Path, device: torch.device, export_team: str) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    algorithm = str(checkpoint.get("algorithm", ""))
    if algorithm in ("object_centric_world_model_sac_flow_selfplay", "cbg_wm_sac_flow_selfplay"):
        config = checkpoint.get("config", {})
        actor_mode = str(checkpoint.get("actor_mode", config.get("actor_mode", "dual")))
        policy = MultiAgentFlowActors(
            int(checkpoint["obs_dim"]),
            int(checkpoint["action_dim"]),
            int(config.get("hidden_dim", 256)),
            actor_mode=actor_mode,
            flow_steps=int(config.get("flow_steps", 3)),
            velocity_scale=float(config.get("flow_velocity_scale", 0.20)),
        ).to(device)
        policy.load_state_dict(checkpoint["actor_state_dict"])
        policy.eval()
        actor = FlowActorOnly(policy, export_team).to(device)
        actor.eval()
        return actor, checkpoint

    raise ValueError(
        f"unsupported checkpoint algorithm {algorithm!r}; "
        "formal export only accepts legacy SAC Flow or CBG-WM checkpoints"
    )


def export_suffix(checkpoint: dict, export_team: str) -> str:
    actor_mode = str(checkpoint.get("actor_mode", checkpoint.get("config", {}).get("actor_mode", "shared")))
    if actor_mode != "dual" or export_team == "auto":
        return ""
    return f"_{export_team}"


def export_torchscript(actor: nn.Module, checkpoint: dict, output_dir: Path, device: torch.device) -> Path:
    example = torch.zeros(1, int(checkpoint["obs_dim"]), dtype=torch.float32, device=device)
    traced = torch.jit.trace(actor, example)
    prefix = "sacflow_tactical_actor"
    output_path = output_dir / f"{prefix}{export_suffix(checkpoint, actor.export_team)}.ts"
    traced.save(str(output_path))
    return output_path


def export_onnx(actor: nn.Module, checkpoint: dict, output_dir: Path, device: torch.device) -> Path:
    example = torch.zeros(1, int(checkpoint["obs_dim"]), dtype=torch.float32, device=device)
    prefix = "sacflow_tactical_actor"
    output_path = output_dir / f"{prefix}{export_suffix(checkpoint, actor.export_team)}.onnx"
    torch.onnx.export(
        actor,
        example,
        output_path,
        input_names=["local_observation"],
        output_names=["tactical_action"],
        dynamic_axes={"local_observation": {0: "batch"}, "tactical_action": {0: "batch"}},
        opset_version=17,
    )
    return output_path


def build_manifest(
    *,
    checkpoint_path: Path,
    export_format: str,
    exported_path: Path | None,
    checkpoint: dict,
    device: torch.device,
    export_team: str,
) -> dict[str, object]:
    return {
        "project": "RoboCup VisionRL",
        "source_checkpoint": str(checkpoint_path),
        "export_format": export_format,
        "algorithm": str(checkpoint.get("algorithm", "object_centric_world_model_sac_flow_selfplay")),
        "actor_mode": str(checkpoint.get("actor_mode", checkpoint.get("config", {}).get("actor_mode", "shared"))),
        "export_team": export_team,
        "exported_actor": str(exported_path) if exported_path is not None else None,
        "device": str(device),
        "obs_dim": int(checkpoint["obs_dim"]),
        "central_obs_dim": int(checkpoint.get("central_obs_dim", 0)),
        "object_state_dim": int(checkpoint.get("object_state_dim", 0)),
        "belief_state_dim": int(checkpoint.get("belief_state_dim", 0)),
        "action_dim": int(checkpoint["action_dim"]),
        "action_labels": list(TACTICAL_ACTION_LABELS),
        "agents": list(checkpoint.get("agents", [])),
        "deployment_contract": {
            "input": f"local_observation[{int(checkpoint['obs_dim'])}]",
            "output": "tactical_action[6] in [-1, 1]",
            "runtime_owner": "rcvrl_behavior",
            "safety_gate": "opponent-target fire gate remains outside the learned actor",
            "planner": "CBG-WM/CVaR MPC remains a runtime component and is not embedded in the actor-only export",
            "ros2_runtime": [
                "rcvrl_behavior",
                "Nav2 NavigateToPose",
                "rcvrl_vision target_detection",
                "rcvrl_shooter services",
                "robot_localization EKF",
            ],
        },
        "training_config": checkpoint.get("config", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Export a trained tactical actor for Sim2Real deployment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../output/policy_export"))
    parser.add_argument("--format", choices=["torchscript", "onnx", "manifest"], default="torchscript")
    parser.add_argument(
        "--team",
        choices=["auto", "yellow", "blue"],
        default="auto",
        help="For dual-actor checkpoints, export an auto-dispatch actor or a fixed team actor.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor, checkpoint = load_actor(args.checkpoint, device, args.team)
    exported_path = None
    if args.format == "torchscript":
        exported_path = export_torchscript(actor, checkpoint, args.output_dir, device)
    elif args.format == "onnx":
        exported_path = export_onnx(actor, checkpoint, args.output_dir, device)

    manifest = build_manifest(
        checkpoint_path=args.checkpoint,
        export_format=args.format,
        exported_path=exported_path,
        checkpoint=checkpoint,
        device=device,
        export_team=args.team,
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "exported_actor": str(exported_path) if exported_path else None}, indent=2))


if __name__ == "__main__":
    main()
