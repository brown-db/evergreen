import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

from experiments.claim_evaluator import Implementation
from experiments.common import EVALUATION_LANGUAGE_MODELS


def run_evaluator(
    module: str,
    impls: list[str],
    lms: list[str],
    eval_sim_filter: bool,
    trial_count: int,
) -> tuple[str, int]:
    start = datetime.now()
    print(f"[START {start:%H:%M:%S}] {module}")

    cmd = [
        sys.executable,
        "-m",
        module,
        "--impls",
        *impls,
        "--lms",
        *lms,
        "--trial_count",
        str(trial_count),
    ]
    if eval_sim_filter:
        cmd.append("--eval_sim_filter")

    result = subprocess.run(cmd)

    end = datetime.now()
    status = "DONE" if result.returncode == 0 else "FAIL"
    print(f"[{status}  {end:%H:%M:%S}] {module} ({(end - start).seconds}s)")

    return module, result.returncode


def main(
    include: list[str],
    exclude: list[str],
    impls: list[str],
    lms: list[str],
    eval_sim_filter: bool,
    trial_count: int,
    max_workers: int,
):
    claim_evaluator_dir = Path("experiments/eval/claim_evaluator")

    if not claim_evaluator_dir.exists():
        print(f"Error: {claim_evaluator_dir} not found. Run from workspace root.")
        sys.exit(1)

    py_files = sorted(
        f for f in claim_evaluator_dir.rglob("*.py") if "__pycache__" not in str(f)
    )
    modules = [str(f).replace("/", ".").removesuffix(".py") for f in py_files]

    if include:
        modules = [m for m in modules if any(p in m for p in include)]
    if exclude:
        modules = [m for m in modules if not any(p in m for p in exclude)]

    if not modules:
        print("No modules matching the given include/exclude patterns")
        sys.exit(1)

    print(f"Running {len(modules)} evaluators:")
    print("  Modules:")
    for m in modules:
        print(f"    - {m}")
    print(f"  Implementations: {impls}")
    print(f"  Models: {lms}")
    print(f"  Trials: {trial_count}")
    print(f"  Workers: {max_workers}")
    if eval_sim_filter:
        print("  Similarity filter evaluation: enabled")
    print()

    total_start = datetime.now()

    run_evaluator_with_impls = partial(
        run_evaluator,
        impls=impls,
        lms=lms,
        eval_sim_filter=eval_sim_filter,
        trial_count=trial_count,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_evaluator_with_impls, modules))

    failed = [m for m, code in results if code != 0]

    print("\n" + "=" * 60)
    print(f"Total time: {(datetime.now() - total_start)}")

    if failed:
        print(f"{len(failed)}/{len(modules)} failed: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(modules)} completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", default=[])
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument(
        "--impls",
        nargs="*",
        choices=[i.value for i in Implementation],
        default=[],
    )
    parser.add_argument(
        "--lms",
        nargs="*",
        choices=EVALUATION_LANGUAGE_MODELS,
        default=[],
    )
    parser.add_argument("--eval_sim_filter", action="store_true")
    parser.add_argument("--trial_count", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    main(
        args.include,
        args.exclude,
        args.impls,
        args.lms,
        args.eval_sim_filter,
        args.trial_count,
        args.max_workers,
    )
