"""Run one stage, or all of them:  python -m recsys.cli <stage> [--set section.key=value ...]"""
import argparse

from recsys.config import load_config

ORDER = ["download", "split", "features", "train_retrieval", "build_index", "train_ranker",
         "evaluate", "ab_replay", "export_triton"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=ORDER + ["all"])
    ap.add_argument("--config", help="path to a config file (default: configs/config.yaml)")
    ap.add_argument("--dataset", help="shortcut for --set data.dataset=...  (ml-25m or ml-latest-small)")
    ap.add_argument("--engine", choices=["spark", "pandas"], help="shortcut for --set data.engine=...")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.engine:
        overrides.append(f"data.engine={args.engine}")
    cfg = load_config(args.config, overrides)

    from recsys.pipeline import stages
    for name in (ORDER if args.stage == "all" else [args.stage]):
        print(f"\n===== stage: {name} =====")
        getattr(stages, f"stage_{name}")(cfg)


if __name__ == "__main__":
    main()
