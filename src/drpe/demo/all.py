from __future__ import annotations

import argparse
import runpy
import sys


def _run(module: str, argv: list[str]) -> None:
    """Run a module as if invoked with `python -m <module> ...`."""
    old_argv = sys.argv
    try:
        sys.argv = [module, *argv]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    p = argparse.ArgumentParser(
        description="DRPE bundle demo: run embedding + ranker demos (safe + blocked) and export artifacts"
    )
    p.add_argument("--export", action="store_true", help="export model cards")
    p.add_argument("--emit-ops", action="store_true", help="emit RecSysOps artifacts")
    args, _ = p.parse_known_args()

    common = []
    if args.export:
        common.append("--export")
    if args.emit_ops:
        common.append("--emit-ops")

    print("\n=== [A] Embedding rollout (normal) ===\n")
    _run("drpe.demo.run", [*common])

    print("\n=== [B] Embedding rollout (forced block) ===\n")
    _run("drpe.demo.run", [*common, "--force-block"])

    print("\n=== [C] Ranker rollout (safe) ===\n")
    _run("drpe.demo.ranker_demo", [*common, "--mode", "safe"])

    print("\n=== [D] Ranker rollout (risky / blocked) ===\n")
    _run("drpe.demo.ranker_demo", [*common, "--mode", "risky"])

    print("\n=== Done. Check artifacts/ for model cards + ops outputs. ===\n")


if __name__ == "__main__":
    main()
