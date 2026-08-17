from __future__ import annotations

import argparse


from .config import (
    DATA_DIR
)
from .render import (
    render_video
)
from .report import (
    make_figures,
    write_report
)
from .train import (
    evaluate,
    train
)

def run_all(args: argparse.Namespace) -> None:
    train_args = argparse.Namespace(**vars(args))
    train(train_args)
    rule_kwargs = {
        "agents_per_team": args.agents_per_team,
        "base_hp": args.base_hp,
        "base_damage": args.base_damage,
        "blue_base_damage_multiplier": args.blue_base_damage_multiplier,
        "capture_rate": args.capture_rate,
        "shield_progress_to_open": args.shield_progress_to_open,
        "contact_radius": args.contact_radius,
        "separation_radius": args.separation_radius,
    }
    eval_args = argparse.Namespace(
        checkpoint=str(DATA_DIR / "policy_checkpoint.json"),
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        max_steps=args.max_steps,
        **rule_kwargs,
    )
    evaluate(eval_args)
    render_args = argparse.Namespace(
        checkpoint=str(DATA_DIR / "policy_checkpoint.json"),
        seed=args.render_seed,
        trace_stride=args.trace_stride,
        fps=args.fps,
        seconds=args.video_seconds,
        gif_seconds=args.gif_seconds,
        gif_fps=args.gif_fps,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        **rule_kwargs,
    )
    render_video(render_args)
    make_figures(args)
    write_report()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Large-scale 50v50 multi-agent battle training/evaluation/rendering")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_train_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=507050)
        p.add_argument("--agents-per-team", type=int, default=50)
        p.add_argument("--generations", type=int, default=80)
        p.add_argument("--population", type=int, default=16)
        p.add_argument("--episodes-per-candidate", type=int, default=2)
        p.add_argument("--probe-episodes", type=int, default=4)
        p.add_argument("--elite-frac", type=float, default=0.25)
        p.add_argument("--sigma", type=float, default=0.55)
        p.add_argument("--min-sigma", type=float, default=0.08)
        p.add_argument("--sigma-decay", type=float, default=0.985)
        p.add_argument("--archive-interval", type=int, default=4)
        p.add_argument("--archive-size", type=int, default=8)
        p.add_argument("--init-checkpoint", default="")
        p.add_argument("--log-interval", type=int, default=5)
        p.add_argument("--max-steps", type=int, default=420)
        p.add_argument("--base-hp", type=float, default=None)
        p.add_argument("--base-damage", type=float, default=None)
        p.add_argument("--blue-base-damage-multiplier", type=float, default=None)
        p.add_argument("--capture-rate", type=float, default=None)
        p.add_argument("--shield-progress-to-open", type=float, default=None)
        p.add_argument("--contact-radius", type=float, default=None)
        p.add_argument("--separation-radius", type=float, default=None)
        p.add_argument("--selection-episodes", type=int, default=24)
        p.add_argument("--verbose", action="store_true")

    p_train = sub.add_parser("train")
    add_train_flags(p_train)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", default=str(DATA_DIR / "policy_checkpoint.json"))
    p_eval.add_argument("--episodes", type=int, default=128)
    p_eval.add_argument("--seed", type=int, default=508000)
    p_eval.add_argument("--agents-per-team", type=int, default=50)
    p_eval.add_argument("--max-steps", type=int, default=420)
    p_eval.add_argument("--base-hp", type=float, default=None)
    p_eval.add_argument("--base-damage", type=float, default=None)
    p_eval.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_eval.add_argument("--capture-rate", type=float, default=None)
    p_eval.add_argument("--shield-progress-to-open", type=float, default=None)
    p_eval.add_argument("--contact-radius", type=float, default=None)
    p_eval.add_argument("--separation-radius", type=float, default=None)
    p_render = sub.add_parser("render")
    p_render.add_argument("--checkpoint", default=str(DATA_DIR / "policy_checkpoint.json"))
    p_render.add_argument("--seed", type=int, default=509000)
    p_render.add_argument("--trace-stride", type=int, default=1)
    p_render.add_argument("--fps", type=int, default=30)
    p_render.add_argument("--seconds", type=float, default=30.0)
    p_render.add_argument("--gif-seconds", type=float, default=12.0)
    p_render.add_argument("--gif-fps", type=int, default=8)
    p_render.add_argument("--width", type=int, default=1920)
    p_render.add_argument("--height", type=int, default=1080)
    p_render.add_argument("--agents-per-team", type=int, default=50)
    p_render.add_argument("--max-steps", type=int, default=420)
    p_render.add_argument("--base-hp", type=float, default=None)
    p_render.add_argument("--base-damage", type=float, default=None)
    p_render.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_render.add_argument("--capture-rate", type=float, default=None)
    p_render.add_argument("--shield-progress-to-open", type=float, default=None)
    p_render.add_argument("--contact-radius", type=float, default=None)
    p_render.add_argument("--separation-radius", type=float, default=None)
    p_fig = sub.add_parser("figures")
    p_fig.add_argument("--agents-per-team", type=int, default=50)
    p_fig.add_argument("--max-steps", type=int, default=420)
    p_fig.add_argument("--base-hp", type=float, default=None)
    p_fig.add_argument("--base-damage", type=float, default=None)
    p_fig.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_fig.add_argument("--capture-rate", type=float, default=None)
    p_fig.add_argument("--shield-progress-to-open", type=float, default=None)
    p_fig.add_argument("--contact-radius", type=float, default=None)
    p_fig.add_argument("--separation-radius", type=float, default=None)
    sub.add_parser("report")
    p_all = sub.add_parser("all")
    add_train_flags(p_all)
    p_all.add_argument("--eval-episodes", type=int, default=128)
    p_all.add_argument("--eval-seed", type=int, default=508000)
    p_all.add_argument("--render-seed", type=int, default=509000)
    p_all.add_argument("--trace-stride", type=int, default=1)
    p_all.add_argument("--fps", type=int, default=30)
    p_all.add_argument("--video-seconds", type=float, default=30.0)
    p_all.add_argument("--gif-seconds", type=float, default=12.0)
    p_all.add_argument("--gif-fps", type=int, default=8)
    p_all.add_argument("--width", type=int, default=1920)
    p_all.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "render":
        render_video(args)
    elif args.cmd == "figures":
        make_figures(args)
    elif args.cmd == "report":
        write_report()
    elif args.cmd == "all":
        run_all(args)
    else:
        raise ValueError(args.cmd)
