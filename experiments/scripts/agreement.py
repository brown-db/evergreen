"""Human-vs-ensemble agreement study for reference (`evg_ref`) labels.

The reference implementation derives ground-truth labels for each per-tuple
semantic operator (LLM filter/map) via a majority vote over a disjoint model
ensemble (`ENSEMBLE_LANGUAGE_MODELS`). To show those labels are trustworthy
(and not merely an artifact of any single model), we draw a small, stratified
sample of per-tuple decisions, have a human label them blind, and report the
human-vs-ensemble agreement.

There are two commands, matching the two phases of the study (a human labels
the tasks in between):

* ``build`` -- discover and de-duplicate the distinct semantic operators
  (operators sharing a dataset and prompt are collapsed so each distinct LLM
  decision is sampled once), recover their per-tuple ensemble decisions from
  the `evg_ref` checkpoints, print the per-operator pool and label counts, then
  draw a stratified, reproducible sample and emit a blind annotation file plus
  a held-out key. Sampling is prefix-nested in the per-operator budget, so
  raising it later only appends tasks; prior human labels are reused by merging
  with the existing annotation file.
* ``score`` -- join the human-labeled tasks against the key and report
  human-vs-ensemble agreement (exact match + Wilson 95% CI + Cohen's kappa),
  overall and broken down by kind and operator.

Run from the workspace root, e.g.::

    python -m experiments.scripts.agreement build
    # ... annotate agreement_tasks.json by hand ...
    python -m experiments.scripts.agreement score
"""

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from evergreen.core.session_context import SessionContext
from evergreen.storage.row_id import RowId
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    SemanticOperator,
    SemanticOperatorKind,
)
from experiments.common import RESULTS_DIR

CLAIM_EVALUATOR_DIR = Path("experiments/eval/claim_evaluator")
AGREEMENT_DIR = RESULTS_DIR / "agreement"
TASKS_FILENAME = "agreement_tasks.json"
KEY_FILENAME = "agreement_key.json"
RESULTS_FILENAME = "agreement_results.json"

DEFAULT_PER_OPERATOR = 4
DEFAULT_SEED = 42


@dataclass
class DistinctOperator:
    """A semantic operator de-duplicated across the claims that use it.

    Multiple claims can apply the exact same LLM decision (same dataset + same
    prompt). Sampling each such decision once avoids pseudo-replication, so the
    agreement numbers reflect distinct operators rather than repeated claims.
    """

    operator_id: str
    kind: SemanticOperatorKind
    dataset_path: str
    text_field: str
    prompt_str: str
    return_type: type[object]
    alias: str | None
    pre_checkpoint_path: Path
    post_checkpoint_path: Path
    # NAMEs of every claim evaluator that uses this operator (representative
    # first); the checkpoints above come from the representative claim.
    claim_names: list[str] = field(default_factory=list)


def label_space(return_type: type[object]) -> list[object] | None:
    """Enumerate the closed label space for a return type, if it has one."""
    if return_type is bool:
        return [True, False]
    if issubclass(return_type, Enum):
        return [member.value for member in return_type]
    return None


def label_value(label: object) -> object:
    """Normalize a stored label to a plain, JSON-friendly value."""
    return label.value if isinstance(label, Enum) else label


def _import_all_evaluator_modules() -> None:
    """Import every claim-evaluator module so its subclass is registered.

    Checkpoints are pickled while each evaluator runs as `__main__`, so any
    enum defined in a claim file (e.g. a categorical map's return type) is
    stored as `__main__.<Enum>`. Mirror those enums into this process's
    `__main__` so `read_pickle` can resolve them.
    """
    main_module = sys.modules["__main__"]
    for path in sorted(CLAIM_EVALUATOR_DIR.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        mod = importlib.import_module(str(path).replace("/", ".").removesuffix(".py"))
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and issubclass(obj, Enum):
                setattr(main_module, name, obj)


def all_evaluator_classes() -> list[type[ClaimEvaluator]]:
    """Every concrete `ClaimEvaluator` subclass, sorted for determinism."""
    _import_all_evaluator_modules()

    classes: set[type[ClaimEvaluator]] = set()
    stack = list(ClaimEvaluator.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if not inspect.isabstract(cls):
            classes.add(cls)

    # `NAME` is not unique across datasets, so disambiguate with the dataset.
    return sorted(classes, key=lambda c: (c.NAME, str(c.AGG_RESULT_PATH)))


def _dedup_key(operator: SemanticOperator, dataset_path: str) -> tuple[object, ...]:
    """Key identifying a distinct LLM decision.

    Maps additionally key on `return_type`: the same prompt text producing a
    different label space is a different decision. Filters are always boolean.
    """
    if operator.kind is SemanticOperatorKind.MAP:
        return (operator.kind, dataset_path, operator.prompt_str, operator.return_type)
    return (operator.kind, dataset_path, operator.prompt_str)


def _operator_id(operator: SemanticOperator, dataset_path: str) -> str:
    """Stable, content-derived id; identical decisions get the same id.

    Hashes the same fields as `_dedup_key` so the id survives reordering and
    addition/removal of claims, keeping task and key files collatable.
    """
    payload = "\x1f".join(
        (
            operator.kind.value,
            dataset_path,
            operator.return_type.__name__,
            operator.prompt_str,
        )
    )
    digest = hashlib.md5(payload.encode()).hexdigest()
    return f"{operator.kind.value}-{digest}"


def discover_operators(
    classes: list[type[ClaimEvaluator]] | None = None,
) -> list[DistinctOperator]:
    """Discover and de-duplicate the semantic operators across all evaluators."""
    if classes is None:
        classes = all_evaluator_classes()

    registry: dict[tuple[object, ...], DistinctOperator] = {}
    ordered: list[DistinctOperator] = []

    for cls in classes:
        evaluator = cls()
        for operator in evaluator.semantic_operators():
            key = _dedup_key(operator, evaluator.dataset_path)
            existing = registry.get(key)
            if existing is not None:
                existing.claim_names.append(evaluator.NAME)
                continue

            distinct = DistinctOperator(
                operator_id=_operator_id(operator, evaluator.dataset_path),
                kind=operator.kind,
                dataset_path=evaluator.dataset_path,
                text_field=evaluator.TEXT_FIELD_NAME,
                prompt_str=operator.prompt_str,
                return_type=operator.return_type,
                alias=operator.alias,
                pre_checkpoint_path=evaluator.reference_checkpoint_path(
                    CheckpointType.PRE_SEM_OP
                ),
                post_checkpoint_path=evaluator.reference_checkpoint_path(
                    CheckpointType.POST_SEM_OP
                ),
                claim_names=[evaluator.NAME],
            )
            registry[key] = distinct
            ordered.append(distinct)

    return ordered


@dataclass(frozen=True)
class CandidateDecision:
    """One per-tuple ensemble decision: the unit a human re-labels blind."""

    operator_id: str
    row_id: RowId
    text: str
    ensemble_label: object


def _require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference checkpoint not found: {path}. Run the evg_ref "
            "implementation for a claim using this operator first."
        )


def load_candidates(operator: DistinctOperator) -> list[CandidateDecision]:
    """Recover every per-tuple ensemble decision for an operator.

    For maps the label is the value in the operator's output column of the
    post-semantic-op checkpoint. For filters the label is membership: a row is
    `True` iff it survived into the post checkpoint (the kept set), `False`
    otherwise. Labels are normalized via `label_value`.
    """
    ctx = SessionContext()

    _require_checkpoint(operator.post_checkpoint_path)
    post_df = ctx.read_pickle(str(operator.post_checkpoint_path))
    post_schema = post_df.schema()
    post_rows = post_df.collect().rows

    if operator.kind is SemanticOperatorKind.MAP:
        assert operator.alias is not None
        text_index = post_schema.index_of(operator.text_field)
        label_index = post_schema.index_of(operator.alias)
        candidates = [
            CandidateDecision(
                operator_id=operator.operator_id,
                row_id=row.id(post_schema),
                text=str(row[text_index]),
                ensemble_label=label_value(row[label_index]),
            )
            for row in post_rows
        ]
    else:
        _require_checkpoint(operator.pre_checkpoint_path)
        pre_df = ctx.read_pickle(str(operator.pre_checkpoint_path))
        pre_schema = pre_df.schema()
        text_index = pre_schema.index_of(operator.text_field)
        kept_row_ids = {row.id(post_schema) for row in post_rows}
        candidates = [
            CandidateDecision(
                operator_id=operator.operator_id,
                row_id=row.id(pre_schema),
                text=str(row[text_index]),
                ensemble_label=row.id(pre_schema) in kept_row_ids,
            )
            for row in pre_df.collect().rows
        ]

    return candidates


def _task_id(operator_id: str, row_id: RowId) -> str:
    """Content-stable task id so prior labels collate across rebuilds."""
    digest = hashlib.md5(repr(row_id).encode()).hexdigest()
    return f"{operator_id}-{digest}"


def _shuffle_key(seed: int, operator_id: str, row_id: RowId) -> str:
    """Deterministic per-candidate sort key (a seeded, stable shuffle)."""
    return hashlib.md5(f"{seed}\x1f{operator_id}\x1f{row_id!r}".encode()).hexdigest()


def stratified_sample(
    operator: DistinctOperator,
    candidates: list[CandidateDecision],
    n: int,
    seed: int,
) -> list[CandidateDecision]:
    """Pick up to `n` candidates, balanced across labels and prefix-nested.

    Each label stratum is deterministically shuffled (by `seed`), then strata
    are visited round-robin. Taking the first `n` is therefore both balanced
    and nested in `n`: a larger budget is a strict superset of a smaller one.
    """
    by_label: dict[object, list[CandidateDecision]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.ensemble_label].append(candidate)
    for members in by_label.values():
        members.sort(key=lambda c: _shuffle_key(seed, operator.operator_id, c.row_id))

    order = sorted(by_label, key=str)
    selected: list[CandidateDecision] = []
    depth = 0
    while len(selected) < n and any(depth < len(by_label[label]) for label in order):
        for label in order:
            if depth < len(by_label[label]):
                selected.append(by_label[label][depth])
                if len(selected) >= n:
                    break
        depth += 1
    return selected


def _cmd_build() -> None:
    out_dir = AGREEMENT_DIR
    tasks_path = out_dir / TASKS_FILENAME
    key_path = out_dir / KEY_FILENAME

    prior_labels = _prior_human_labels(tasks_path)
    prior_ensemble = _prior_ensemble_labels(key_path)

    operators = discover_operators()
    tasks: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    skipped: list[DistinctOperator] = []
    reused = 0
    changed: list[str] = []

    for op in operators:
        try:
            candidates = load_candidates(op)
        except FileNotFoundError:
            skipped.append(op)
            continue

        sample = stratified_sample(op, candidates, DEFAULT_PER_OPERATOR, DEFAULT_SEED)
        counts = Counter(c.ensemble_label for c in candidates)
        labels = label_space(op.return_type)
        label_str = "/".join(str(v) for v in labels) if labels else "<free-form>"
        print(
            f"[{op.operator_id}] {op.kind.value}  {Path(op.dataset_path).name}"
            f"  {op.return_type.__name__} ({label_str})"
            f"  pool={len(candidates)} sampled={len(sample)}"
        )
        for label, count in sorted(counts.items(), key=lambda kv: str(kv[0])):
            taken = sum(1 for c in sample if c.ensemble_label == label)
            print(f"    {label}: {taken}/{count}")

        for candidate in sample:
            task_id = _task_id(op.operator_id, candidate.row_id)
            human_label = prior_labels.get(task_id)
            if human_label is not None:
                reused += 1
                if (
                    task_id in prior_ensemble
                    and prior_ensemble[task_id] != candidate.ensemble_label
                ):
                    changed.append(task_id)

            tasks.append(
                {
                    "task_id": task_id,
                    "operator_id": op.operator_id,
                    "kind": op.kind.value,
                    "prompt": op.prompt_str,
                    "text": candidate.text,
                    "label_space": label_space(op.return_type),
                    "human_label": human_label,
                }
            )
            keys.append(
                {
                    "task_id": task_id,
                    "operator_id": op.operator_id,
                    "ensemble_label": candidate.ensemble_label,
                    "row_id": list(candidate.row_id),
                }
            )

    selected_ids = {task["task_id"] for task in tasks}
    dropped = sorted(
        task_id
        for task_id, label in prior_labels.items()
        if label is not None and task_id not in selected_ids
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(tasks, indent=2) + "\n")
    key_path.write_text(json.dumps(keys, indent=2) + "\n")

    print()
    print(f"Wrote {len(tasks)} tasks from {len(operators) - len(skipped)} operators.")
    print(f"  tasks: {tasks_path}")
    print(f"  key  : {key_path}")
    if reused:
        print(f"Reused {reused} prior human label(s).")
    if changed:
        print(
            f"WARNING: {len(changed)} reused label(s) now disagree with a changed "
            f"ensemble label; re-check: {changed}"
        )
    if dropped:
        print(
            f"WARNING: {len(dropped)} previously-labeled task(s) are no longer "
            f"sampled (labels lost): {dropped}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} operator(s) with missing checkpoints:")
        for op in skipped:
            print(f"    [{op.operator_id}] {op.post_checkpoint_path}")


def _prior_human_labels(tasks_path: Path) -> dict[str, object]:
    if not tasks_path.exists():
        return {}
    return {
        task["task_id"]: task.get("human_label")
        for task in json.loads(tasks_path.read_text())
    }


def _prior_ensemble_labels(key_path: Path) -> dict[str, object]:
    if not key_path.exists():
        return {}
    return {
        entry["task_id"]: entry["ensemble_label"]
        for entry in json.loads(key_path.read_text())
    }


def _wilson_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (center - half, center + half)


def _cohens_kappa(pairs: list[tuple[object, object]]) -> float | None:
    """Cohen's kappa for two raters; `None` when chance agreement is degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    categories = {a for a, _ in pairs} | {b for _, b in pairs}
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum(
        (sum(1 for a, _ in pairs if a == c) / n)
        * (sum(1 for _, b in pairs if b == c) / n)
        for c in categories
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class AgreementMetrics:
    n: int
    agree: int
    agreement: float | None
    wilson_95ci: tuple[float, float]
    cohens_kappa: float | None


def _agreement_metrics(pairs: list[tuple[object, object]]) -> AgreementMetrics:
    """Exact-match agreement, Wilson 95% CI, and Cohen's kappa for label pairs."""
    n = len(pairs)
    agree = sum(1 for human, ensemble in pairs if human == ensemble)
    return AgreementMetrics(
        n=n,
        agree=agree,
        agreement=agree / n if n else None,
        wilson_95ci=_wilson_interval(agree, n),
        cohens_kappa=_cohens_kappa(pairs),
    )


def _format_metrics(metrics: AgreementMetrics, show_ci: bool = True) -> str:
    if metrics.n == 0:
        return "no labeled tasks"
    kappa_str = "n/a" if metrics.cohens_kappa is None else f"{metrics.cohens_kappa:.3f}"
    parts = [f"agreement={metrics.agreement:.3f} ({metrics.agree}/{metrics.n})"]
    if show_ci:
        low, high = metrics.wilson_95ci
        parts.append(f"95% CI [{low:.3f}, {high:.3f}]")
    parts.append(f"kappa={kappa_str}")
    return "  ".join(parts)


@dataclass(frozen=True)
class _ScoredTask:
    operator_id: str
    kind: str
    human: object
    ensemble: object


def _cmd_score() -> None:
    out_dir = AGREEMENT_DIR
    tasks_path = out_dir / TASKS_FILENAME
    key_path = out_dir / KEY_FILENAME

    if not tasks_path.exists() or not key_path.exists():
        raise FileNotFoundError(
            f"Expected {TASKS_FILENAME} and {KEY_FILENAME} in {out_dir}; run "
            "`build` and annotate the tasks first."
        )

    tasks = json.loads(tasks_path.read_text())
    ensemble_by_id = {
        entry["task_id"]: entry["ensemble_label"]
        for entry in json.loads(key_path.read_text())
    }

    scored: list[_ScoredTask] = []
    unlabeled = 0
    orphan = 0
    for task in tasks:
        if task.get("human_label") is None:
            unlabeled += 1
            continue
        if task["task_id"] not in ensemble_by_id:
            orphan += 1
            continue
        scored.append(
            _ScoredTask(
                operator_id=task["operator_id"],
                kind=task["kind"],
                human=task["human_label"],
                ensemble=ensemble_by_id[task["task_id"]],
            )
        )

    def metrics_for(subset: list[_ScoredTask]) -> AgreementMetrics:
        return _agreement_metrics([(t.human, t.ensemble) for t in subset])

    overall = metrics_for(scored)
    by_kind = {
        kind: metrics_for([t for t in scored if t.kind == kind])
        for kind in sorted({t.kind for t in scored})
    }
    by_operator = {
        operator_id: metrics_for([t for t in scored if t.operator_id == operator_id])
        for operator_id in sorted({t.operator_id for t in scored})
    }

    results = {
        "labeled": len(scored),
        "unlabeled": unlabeled,
        "overall": asdict(overall),
        "by_kind": {kind: asdict(m) for kind, m in by_kind.items()},
        "by_operator": {op: asdict(m) for op, m in by_operator.items()},
    }

    print(f"Labeled {len(scored)} task(s); {unlabeled} unlabeled.")
    if orphan:
        print(f"WARNING: {orphan} labeled task(s) had no key entry (ignored).")
    print()
    print(f"OVERALL   {_format_metrics(overall)}")
    print("\nBy kind:")
    for kind, metrics in by_kind.items():
        print(f"  {kind:7s} {_format_metrics(metrics)}")
    print("\nBy operator:")
    for operator_id, metrics in by_operator.items():
        print(f"  [{operator_id}] {_format_metrics(metrics, show_ci=False)}")
    print(
        "\nNote: kappa is n/a when a group's labels are all one class (common at "
        "small per-operator n); it is most meaningful overall and per-kind."
    )

    results_path = out_dir / RESULTS_FILENAME
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "build",
        help="Sample a stratified annotation task set and held-out key.",
    )
    subparsers.add_parser(
        "score",
        help="Report human-vs-ensemble agreement from the annotated tasks.",
    )

    args = parser.parse_args()
    if args.command == "build":
        _cmd_build()
    else:
        _cmd_score()


if __name__ == "__main__":
    main()
