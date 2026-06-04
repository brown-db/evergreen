import json
import shutil
from enum import Enum, auto
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.common import MODEL_PRICING

RESULTS_DIR = Path("experiments/results/claim_evaluator")
FIGURES_DIR = Path("experiments/figures")
PAPER_DIR = Path("paper")
PAPER_FIGURES_DIR = PAPER_DIR / "figures"

CLAIM_TO_CLAIM_ID = {
    "johns_roast_pork\nsummarize\ncardinal_claim_1": "C1",
    "johns_roast_pork\nsummarize\nexistential_claim_1": "C2",
    "johns_roast_pork\nsummarize\nproportional_claim_1": "C3",
    "johns_roast_pork\nsummarize\nuniversal_claim_1": "C4",
    "mcdonalds_mo\ncompare\nexistential_universal_claim_1": "C5",
    "mcdonalds_mo\ncompare\nproportional_cardinal_claim_1": "C6",
    "mcdonalds_mo\ncompare\nproportional_existential_claim_1": "C7",
    "mcdonalds_mo\ncompare\nuniversal_cardinal_claim_1": "C8",
    "mcdonalds_mo\nrank\nordinal_claim_1": "C9",
    "mcdonalds_mo\nrank\nordinal_claim_2": "C10",
    "mcdonalds_mo\nrank\nordinal_claim_3": "C11",
    "mcdonalds_mo\nrank\nordinal_claim_4": "C12",
    "village_whiskey\nsummarize\ncardinal_claim_1": "C13",
    "village_whiskey\nsummarize\nexistential_claim_1": "C14",
    "village_whiskey\nsummarize\nproportional_claim_1": "C15",
    "village_whiskey\nsummarize\nuniversal_claim_1": "C16",
}

CLAIM_ID_TO_CLAIM = {v: k for k, v in CLAIM_TO_CLAIM_ID.items()}

IMPL_MODEL_LABELS = {
    "base_rm_claude-opus-4-6": "base_rm Claude Opus 4.6",
    "base_rm_claude-sonnet-4-6": "base_rm Claude Sonnet 4.6",
    "base_rm_claude-haiku-4-5": "base_rm Claude Haiku 4.5",
    "rag_agent_claude-opus-4-6": "rag_agent Claude Opus 4.6",
    "rag_agent_claude-sonnet-4-6": "rag_agent Claude Sonnet 4.6",
    "rag_agent_claude-haiku-4-5": "rag_agent Claude Haiku 4.5",
    "evg_unopt_claude-opus-4-6": "evg_unopt Claude Opus 4.6",
    "evg_opt_claude-opus-4-6": "evg_opt Claude Opus 4.6",
    "evg_opt_claude-sonnet-4-6": "evg_opt Claude Sonnet 4.6",
    "evg_opt_claude-haiku-4-5": "evg_opt Claude Haiku 4.5",
    "evg_opt_llama4-maverick": "evg_opt Llama 4 Maverick",
    "evg_opt_llama4-scout": "evg_opt Llama 4 Scout",
    "evg_opt_llama3.1-8b": "evg_opt Llama 3.1 8B",
}

IMPLEMENTATIONS = ["base_rm", "rag_agent", "evg_unopt", "evg_opt"]

_MODEL_LABELS = {
    "claude-opus-4-6": "Opus 4.6",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-haiku-4-5": "Haiku 4.5",
    "llama4-maverick": "Maverick",
    "llama4-scout": "Scout",
    "llama3.1-8b": "8B",
}

ABLATION_MODEL = "claude-haiku-4-5"

_ABLATION_STEP_LABELS = {
    "evg_opt": "All Opt",
    "evg_abl_no_es": "ES",
    "evg_abl_no_rs": "RS",
    "evg_abl_no_est": "ECS",
    "evg_abl_no_fus": "OF",
    "evg_abl_no_sf": "SF",
    "evg_abl_no_cache": "PC",
}

ABLATION_IMPLEMENTATIONS = list(_ABLATION_STEP_LABELS)

ABLATION_LABEL_MAP = {
    f"{impl}_{ABLATION_MODEL}": label for impl, label in _ABLATION_STEP_LABELS.items()
}

ABLATION_APPLICABLE_CLAIMS = {
    "evg_abl_no_es": (
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C13",
        "C14",
        "C15",
        "C16",
    ),
    "evg_abl_no_rs": ("C2", "C6", "C7", "C8", "C13", "C14"),
    "evg_abl_no_est": (
        "C1",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C15",
        "C16",
    ),
    "evg_abl_no_fus": ("C3", "C9", "C10", "C11", "C12", "C13", "C16"),
    "evg_abl_no_sf": ("C3", "C9", "C10", "C11", "C12", "C13", "C16"),
    "evg_abl_no_cache": ("C9", "C10", "C11", "C12"),
}


FIGURE_SIZE = (40, 5)


class FormatType(Enum):
    TOKENS = auto()
    COST = auto()
    LATENCY = auto()
    SCORE = auto()
    COUNT = auto()
    MULTIPLIER = auto()


def compute_token_costs(
    language_model: str, lm_metrics: list[dict[str, Any]]
) -> tuple[float, float]:
    pricing = MODEL_PRICING[language_model]

    input_token_count = sum(m["input_token_count"] for m in lm_metrics)
    output_token_count = sum(m["output_token_count"] for m in lm_metrics)
    cache_creation_input_token_count = sum(
        m.get("cache_creation_input_token_count", 0) for m in lm_metrics
    )
    cache_read_input_token_count = sum(
        m.get("cache_read_input_token_count", 0) for m in lm_metrics
    )

    input_token_cost = (
        (
            input_token_count
            + cache_creation_input_token_count
            + cache_read_input_token_count
        )
        * pricing["input"]
        / 1e6
    )
    output_token_cost = output_token_count * pricing["output"] / 1e6

    return input_token_cost, output_token_cost


def get_impl_models(
    df: pd.DataFrame,
    impls: list[str] | None = None,
    labels: dict[str, str] | None = None,
) -> list[str]:
    if impls is None:
        impls = IMPLEMENTATIONS
    label_map = labels if labels is not None else IMPL_MODEL_LABELS
    impl_models: list[str] = []
    for impl in impls:
        impl_df = df[df["implementation"] == impl]
        # Order models consistently based on MODEL_PRICING key order
        for model in MODEL_PRICING:
            key = f"{impl}_{model}"
            if key in impl_df["impl_model"].values and key in label_map:
                impl_models.append(key)
    return impl_models


def load_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for file_path in RESULTS_DIR.rglob("*.json"):
        with open(file_path) as file:
            data = json.load(file)

        if "evaluation_result" not in data:
            continue

        metadata = data["metadata"]
        evaluation_result = data["evaluation_result"]
        query_metrics = evaluation_result["query_metrics"]

        exec_lm_metrics = query_metrics["language_model_metrics"]
        opt_lm_metrics = query_metrics["optimizer_language_model_metrics"]

        model_name = exec_lm_metrics[0]["model_name"]

        exec_input_token_cost, exec_output_token_cost = compute_token_costs(
            model_name, exec_lm_metrics
        )
        opt_input_token_cost, opt_output_token_cost = compute_token_costs(
            "claude-opus-4-6", opt_lm_metrics
        )

        input_token_cost = exec_input_token_cost + opt_input_token_cost
        output_token_cost = exec_output_token_cost + opt_output_token_cost
        total_token_cost = input_token_cost + output_token_cost

        lm_metrics = exec_lm_metrics + opt_lm_metrics

        lm_call_count = sum(
            m["prompt_count"] - m["prompt_cache_hit_count"] for m in lm_metrics
        )
        input_token_count = sum(m["input_token_count"] for m in lm_metrics)
        output_token_count = sum(m["output_token_count"] for m in lm_metrics)

        latency = query_metrics["planning_latency"] + query_metrics["execution_latency"]

        # Parse path: .../claim_evaluator/{source}/{dataset}/{task}/{claim}/
        path_parts = file_path.relative_to(RESULTS_DIR).parts

        impl = metadata["implementation"]

        row = {
            "claim": f"{path_parts[1]}\n{path_parts[2]}\n{metadata['name']}",
            "implementation": impl,
            "model": model_name,
            "trial_id": metadata["trial_id"],
            "impl_model": f"{impl}_{model_name}",
            "verification_result": evaluation_result["verification_result"],
            "input_token_cost": input_token_cost,
            "output_token_cost": output_token_cost,
            "total_token_cost": total_token_cost,
            "lm_call_count": lm_call_count,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
        }

        filter_metrics = evaluation_result["filter_metrics"]
        map_metrics = evaluation_result["map_metrics"]
        prov_tokens = evaluation_result["prov_tokens"]

        if prov_tokens:
            row["prov_valid_count"] = sum(t["is_valid"] for t in prov_tokens)
            row["prov_total"] = len(prov_tokens)
            row["provenance_precision"] = row["prov_valid_count"] / row["prov_total"]

        if filter_metrics:
            # Raw counts for aggregate computation
            row["filter_tp"] = filter_metrics["tp"]
            row["filter_fp"] = filter_metrics["fp"]
            row["filter_fn"] = filter_metrics["fn"]
            # Computed metrics for per-claim view
            row["filter_precision"] = filter_metrics["precision"]
            row["filter_recall"] = filter_metrics["recall"]
            row["filter_f1"] = filter_metrics["f1_score"]

        if map_metrics:
            # Raw counts for aggregate computation
            row["map_correct"] = sum(map_metrics["correct_by_column"].values())
            row["map_total"] = map_metrics["total_rows"] * len(
                map_metrics["correct_by_column"]
            )
            # Computed metric for per-claim view
            row["map_accuracy"] = map_metrics["accuracy"]

        rows.append(row)

    return pd.DataFrame(rows)


def format_value(value: float, format_type: FormatType) -> str:
    match format_type:
        case FormatType.TOKENS:
            if value >= 1e6:
                return f"{value / 1e6:.1f}M"
            if value >= 1e3:
                return f"{value / 1e3:.0f}k"
            return f"{value:.0f}"
        case FormatType.COST:
            return f"${value:.2f}" if value >= 1 else f"${value:.3f}"
        case FormatType.LATENCY:
            return f"{value / 1e3:.1f}k" if value >= 1e3 else f"{value:.0f}"
        case FormatType.SCORE:
            return f"{value:.2f}"
        case FormatType.COUNT:
            return f"{value:.0f}"
        case FormatType.MULTIPLIER:
            return f"{value:.1f}"


def add_bar_labels(axis: Any, format_type: FormatType, rotation: int = 0) -> None:
    for container in axis.containers:
        labels = [format_value(bar.get_height(), format_type) for bar in container]
        axis.bar_label(container, labels=labels, rotation=rotation)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


def save_table(
    rows_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    filename: str,
    format_type: FormatType,
    pivot_col: str | None = None,
) -> None:
    summary = (
        rows_df.groupby(group_cols)[value_col].agg(["mean", "min", "max"]).reset_index()
    )
    summary["formatted"] = [
        f"{format_value(row['mean'], format_type)} "
        f"[{format_value(row['min'], format_type)}, "
        f"{format_value(row['max'], format_type)}]"
        for _, row in summary.iterrows()
    ]
    if pivot_col:
        table = summary.pivot(
            index=[c for c in group_cols if c != pivot_col],
            columns=pivot_col,
            values="formatted",
        ).reset_index()
    else:
        table = summary[group_cols + ["formatted"]]

    table.to_csv(FIGURES_DIR / f"{filename}.csv", index=False)


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def plot_precision_recall_f1(
    rows: list[dict[str, object]], ylabel: str, order: list[str], filename: str
) -> None:
    _, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        data=pd.DataFrame(rows),
        x="impl_model",
        y="value",
        hue="metric",
        ax=axis,
        order=order,
        errorbar=("pi", 100),
        err_kws={"alpha": 0.5},
    )
    axis.set(xlabel="implementation", ylabel=ylabel)  # type: ignore
    add_bar_labels(axis, FormatType.SCORE)
    save_figure(filename)


def plot_verification_quality(
    df: pd.DataFrame,
    impls: list[str] | None = None,
    filename: str = "verification_quality",
    labels: dict[str, str] | None = None,
) -> None:
    """Plot verification precision/recall/f1 compared to evg_ref."""
    label_map = labels if labels is not None else IMPL_MODEL_LABELS
    reference = df[df["implementation"] == "evg_ref"].set_index("claim")[
        "verification_result"
    ]
    rows: list[dict[str, object]] = []

    impl_models = get_impl_models(df, impls=impls, labels=label_map)
    for impl_model in impl_models:
        impl_df = df[df["impl_model"] == impl_model]
        label = label_map[impl_model]
        for trial_id in impl_df["trial_id"].unique():
            trial_df = impl_df[impl_df["trial_id"] == int(trial_id)].set_index("claim")[
                "verification_result"
            ]
            common = reference.index.intersection(trial_df.index)
            ref, pred = reference.loc[common], trial_df.loc[common]

            tp = (~ref & ~pred).sum()
            fp = (ref & ~pred).sum()
            fn = (~ref & pred).sum()

            precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
            accuracy = (ref == pred).sum() / len(common) if len(common) else 0

            for metric, value in [
                ("precision", precision),
                ("recall", recall),
                ("f1", f1),
                ("accuracy", accuracy),
            ]:
                rows.append({"impl_model": label, "metric": metric, "value": value})

    ordered = [label_map[m] for m in impl_models]
    plot_precision_recall_f1(
        rows,
        ylabel="verification score",
        order=ordered,
        filename=filename,
    )
    save_table(
        pd.DataFrame(rows),
        "value",
        ["impl_model", "metric"],
        filename,
        FormatType.SCORE,
        pivot_col="metric",
    )


def plot_filter_quality_aggregated(df: pd.DataFrame) -> None:
    """Plot filter precision/recall/f1 aggregated across all claims."""
    filtered = df[(df["implementation"] != "evg_ref") & df["filter_tp"].notna()]

    agg = (
        filtered.groupby(["impl_model", "trial_id"])[
            ["filter_tp", "filter_fp", "filter_fn"]
        ]
        .sum()
        .reset_index()
    )
    rows: list[dict[str, object]] = []

    for _, row_data in agg.iterrows():
        impl_model = row_data["impl_model"]
        tp = int(row_data["filter_tp"])
        fp = int(row_data["filter_fp"])
        fn = int(row_data["filter_fn"])
        precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

        for metric, value in [("precision", precision), ("recall", recall), ("f1", f1)]:
            rows.append({"impl_model": impl_model, "metric": metric, "value": value})

    plot_precision_recall_f1(
        rows,
        ylabel="filter score",
        order=get_impl_models(df, impls=["evg_unopt", "evg_opt"]),
        filename="filter_quality_aggregated",
    )
    save_table(
        pd.DataFrame(rows),
        "value",
        ["impl_model", "metric"],
        "filter_quality_aggregated",
        FormatType.SCORE,
        pivot_col="metric",
    )


def plot_map_accuracy_aggregated(df: pd.DataFrame) -> None:
    """Plot map accuracy aggregated across all claims."""
    filtered = df[(df["implementation"] != "evg_ref") & df["map_total"].notna()]

    agg = (
        filtered.groupby(["impl_model", "trial_id"])[["map_correct", "map_total"]]
        .sum()
        .reset_index()
    )
    rows: list[dict[str, object]] = []

    for _, row_data in agg.iterrows():
        impl_model = row_data["impl_model"]
        correct = row_data["map_correct"]
        total = row_data["map_total"]
        accuracy = correct / total if total else 0  # type: ignore
        rows.append({"impl_model": impl_model, "accuracy": accuracy})

    _, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        data=pd.DataFrame(rows),
        x="impl_model",
        y="accuracy",
        ax=axis,
        order=get_impl_models(df, impls=["evg_unopt", "evg_opt"]),
        errorbar=("pi", 100),
        err_kws={"alpha": 0.5},
    )
    axis.set(xlabel="implementation", ylabel="accuracy")  # type: ignore
    add_bar_labels(axis, FormatType.SCORE)
    save_figure("map_accuracy_aggregated")
    save_table(
        pd.DataFrame(rows),
        "accuracy",
        ["impl_model"],
        "map_accuracy_aggregated",
        FormatType.SCORE,
    )


def plot_verification_heatmap(df: pd.DataFrame) -> None:
    """Plot heatmap showing correctness of each implementation vs reference.
    Shows mean correctness across trials.
    """
    reference = df[df["implementation"] == "evg_ref"].set_index("claim")[
        "verification_result"
    ]

    impl_models = get_impl_models(df)
    rows: list[dict[str, object]] = []
    for impl_model in impl_models:
        impl_df = df[df["impl_model"] == impl_model]
        for _, row_data in impl_df.iterrows():
            claim = row_data["claim"]
            result = row_data["verification_result"]
            rows.append(
                {
                    "claim": claim,
                    "impl_model": impl_model,
                    "correct": float(result == reference.loc[str(claim)]),
                }
            )

    correctness_df = pd.DataFrame(rows)
    # Average correctness across trials (1.0 = all correct, 0.0 = none correct)
    pivot = correctness_df.groupby(["claim", "impl_model"])["correct"].mean().unstack()
    # Reorder columns
    pivot = pivot[[c for c in impl_models if c in pivot.columns]]

    pivot.index = [CLAIM_TO_CLAIM_ID[c] for c in pivot.index]
    pivot.columns = [IMPL_MODEL_LABELS[c] for c in pivot.columns]

    _, axis = plt.subplots(figsize=(10, 3))
    sns.heatmap(
        pivot.T,
        annot=True,
        ax=axis,
        vmin=0,
        vmax=1,
        fmt=".2f",
        cbar=False,
    )  # type: ignore
    axis.set(xlabel="Claim", ylabel="Implementation")  # type: ignore
    save_figure("verification_result")


def plot_mean_aggregated(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    format_type: FormatType,
    impls: list[str] | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    """Plot average of a metric per claim across implementations."""
    label_map = labels if labels is not None else IMPL_MODEL_LABELS
    impl_models = get_impl_models(df, impls=impls, labels=label_map)
    filtered = df[df["impl_model"].isin(impl_models)]

    agg = filtered.groupby(["impl_model", "trial_id"])[column].mean().reset_index()
    agg.columns = ["impl_model", "trial_id", column]
    agg["impl_model"] = agg["impl_model"].map(label_map)
    ordered = [label_map[m] for m in impl_models]
    agg["impl_model"] = pd.Categorical(
        agg["impl_model"], categories=ordered, ordered=True
    )
    agg = agg.sort_values("impl_model")

    _, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        data=agg,
        x="impl_model",
        y=column,
        ax=axis,
        errorbar=("pi", 100),
        err_kws={"alpha": 0.5},
    )
    axis.set(xlabel="implementation", ylabel=ylabel)  # type: ignore
    add_bar_labels(axis, format_type)
    save_figure(filename)
    save_table(agg, column, ["impl_model"], filename, format_type)


def plot_provenance_precision_aggregated(df: pd.DataFrame) -> None:
    """Plot provenance precision micro-averaged across all claims."""
    filtered = df[(df["implementation"] != "evg_ref") & df["prov_total"].notna()]

    agg = (
        filtered.groupby(["impl_model", "trial_id"])[["prov_valid_count", "prov_total"]]
        .sum()
        .reset_index()
    )
    agg["provenance_precision"] = agg["prov_valid_count"] / agg["prov_total"]

    impl_models = get_impl_models(df, impls=["evg_unopt", "evg_opt"])
    agg = agg[agg["impl_model"].isin(impl_models)]
    agg["impl_model"] = pd.Categorical(
        agg["impl_model"], categories=impl_models, ordered=True
    )
    agg = agg.sort_values("impl_model")

    _, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        data=agg,
        x="impl_model",
        y="provenance_precision",
        ax=axis,
        errorbar=("pi", 100),
        err_kws={"alpha": 0.5},
    )
    axis.set(xlabel="implementation", ylabel="provenance precision")  # type: ignore
    add_bar_labels(axis, FormatType.SCORE)
    save_figure("provenance_precision_aggregated")
    save_table(
        agg,
        "provenance_precision",
        ["impl_model"],
        "provenance_precision_aggregated",
        FormatType.SCORE,
    )


def plot_metric(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    format_type: FormatType,
    log_scale: bool = False,
    include_ref: bool = False,
    impls: list[str] | None = None,
) -> None:
    """Plot a metric per claim grouped by impl_model."""
    filtered = df[df[column].notna()]
    if not include_ref:
        filtered = filtered[filtered["implementation"] != "evg_ref"]

    impl_models = get_impl_models(df, impls)

    _, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        data=filtered,
        x="claim",
        y=column,
        hue="impl_model",
        hue_order=impl_models,
        ax=axis,
        errorbar=("pi", 100),
        err_kws={"alpha": 0.5},
    )
    axis.set(ylabel=ylabel, yscale="symlog" if log_scale else "linear")  # type: ignore
    axis.legend(title="implementation")  # type: ignore
    add_bar_labels(axis, format_type, rotation=30)
    save_figure(filename)


def plot_ablation_quality(df: pd.DataFrame) -> None:
    """Save quality metric changes for all ablation variants."""

    def _quality(impl_df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
        rows_q: list[dict[str, float]] = []
        for tid in impl_df["trial_id"].unique():
            trial = impl_df[impl_df["trial_id"] == tid].set_index("claim")[
                "verification_result"
            ]
            common = ref.index.intersection(trial.index)
            if not len(common):
                rows_q.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
                continue
            r, p = ref.loc[common], trial.loc[common]
            tp = (~r & ~p).sum()
            fp = (r & ~p).sum()
            fn = (~r & p).sum()
            precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
            rows_q.append({"precision": precision, "recall": recall, "f1": f1})
        return pd.DataFrame(rows_q)

    ablation_df = df[
        (df["model"] == ABLATION_MODEL) | (df["implementation"] == "evg_ref")
    ]
    reference = ablation_df[ablation_df["implementation"] == "evg_ref"].set_index(
        "claim"
    )["verification_result"]
    baseline_key = f"evg_opt_{ABLATION_MODEL}"

    rows: list[dict[str, object]] = []

    for impl, label in _ABLATION_STEP_LABELS.items():
        if impl == "evg_opt":
            continue

        abl_key = f"{impl}_{ABLATION_MODEL}"
        claim_ids = ABLATION_APPLICABLE_CLAIMS[impl]
        claims = {CLAIM_ID_TO_CLAIM[c] for c in claim_ids}
        sub = ablation_df[ablation_df["claim"].isin(claims)]

        base = sub[sub["impl_model"] == baseline_key]
        abl = sub[sub["impl_model"] == abl_key]
        ref = reference[reference.index.isin(claims)]

        base_q = _quality(base, ref)
        abl_q = _quality(abl, ref)

        for metric_col, metric_label in [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]:
            base_val = base_q[metric_col].mean()
            abl_val = abl_q[metric_col].mean()
            rows.append(
                {
                    "Implementation": label,
                    "Metric": metric_label,
                    "value": f"{base_val:.3f}",
                }
            )
            rows.append(
                {
                    "Implementation": label,
                    "Metric": f"Δ {metric_label}",
                    "value": f"{abl_val - base_val:+.3f}",
                }
            )

    result = pd.DataFrame(rows)
    label_order = [v for k, v in _ABLATION_STEP_LABELS.items() if k != "evg_opt"]
    result["Implementation"] = pd.Categorical(
        result["Implementation"], categories=label_order, ordered=True
    )
    table = result.pivot(
        index="Implementation",
        columns="Metric",
        values="value",
    ).reset_index()
    table.columns.name = None
    claims_map = {
        label: ", ".join(
            sorted(ABLATION_APPLICABLE_CLAIMS[impl], key=lambda c: int(c[1:]))
        )
        for impl, label in _ABLATION_STEP_LABELS.items()
        if impl != "evg_opt"
    }
    table["Claims"] = table["Implementation"].map(claims_map)
    table = table[
        [
            "Implementation",
            "Claims",
            "Precision",
            "Δ Precision",
            "Recall",
            "Δ Recall",
            "F1",
            "Δ F1",
        ]
    ]
    table.to_csv(FIGURES_DIR / "ablation_quality.csv", index=False)


def plot_ablation_cost_and_latency(df: pd.DataFrame) -> None:
    """Plot cost and latency multipliers (No X / All Opt) over applicable claims."""
    ablation_df = df[
        (df["model"] == ABLATION_MODEL) | (df["implementation"] == "evg_ref")
    ]
    baseline_key = f"evg_opt_{ABLATION_MODEL}"
    label_order = [v for k, v in _ABLATION_STEP_LABELS.items() if k != "evg_opt"]

    _, (ax_cost, ax_latency) = plt.subplots(1, 2, figsize=(5, 2.5))

    for metric, col_name, axis in [
        ("Cost", "total_token_cost", ax_cost),
        ("Latency", "latency", ax_latency),
    ]:
        rows: list[dict[str, object]] = []

        for impl, label in _ABLATION_STEP_LABELS.items():
            if impl == "evg_opt":
                continue

            abl_key = f"{impl}_{ABLATION_MODEL}"
            claim_ids = ABLATION_APPLICABLE_CLAIMS[impl]
            claims = {CLAIM_ID_TO_CLAIM[c] for c in claim_ids}
            sub = ablation_df[ablation_df["claim"].isin(claims)]

            base = sub[sub["impl_model"] == baseline_key]
            abl = sub[sub["impl_model"] == abl_key]

            base_means = base.groupby("trial_id")[col_name].mean()
            abl_means = abl.groupby("trial_id")[col_name].mean()
            multiplier = abl_means / base_means

            for trial_id, val in multiplier.items():
                rows.append(
                    {
                        "Implementation": label,
                        "trial_id": trial_id,
                        col_name: val,
                    }
                )

        agg = pd.DataFrame(rows)
        agg["Implementation"] = pd.Categorical(
            agg["Implementation"], categories=label_order, ordered=True
        )
        agg = agg.sort_values("Implementation")

        sns.barplot(
            data=agg,
            x="Implementation",
            y=col_name,
            ax=axis,
            errorbar=("pi", 100),
            err_kws={"alpha": 0.5},
        )
        axis.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
        axis.set(xlabel="Implementation", ylabel=f"{metric} Multiplier")
        axis.tick_params(axis="x", rotation=45)
        add_bar_labels(axis, FormatType.MULTIPLIER)

    save_figure("ablation_cost_and_latency")


def load_sim_filter_results() -> pd.DataFrame:
    """Load sim_filter_sensitivity_analysis.json files into a DataFrame."""
    rows: list[dict[str, Any]] = []
    for file_path in RESULTS_DIR.rglob("sim_filter_sensitivity_analysis.json"):
        with open(file_path) as f:
            data = json.load(f)
        metadata = data["metadata"]
        path_parts = file_path.relative_to(RESULTS_DIR).parts
        claim_key = f"{path_parts[1]}\n{path_parts[2]}\n{metadata['name']}"
        claim_id = CLAIM_TO_CLAIM_ID[claim_key]
        for result in data["results"]:
            gt_count = result["ground_truth_selected_row_count"]
            total_count = result["total_row_count"]
            recalls = [t["tp"] / gt_count for t in result["trials"]]
            filter_rates = [
                1 - t["selected_count"] / total_count for t in result["trials"]
            ]
            rows.append(
                {
                    "claim_id": claim_id,
                    "threshold": result["threshold"],
                    "recall_mean": np.mean(recalls),
                    "recall_min": np.min(recalls),
                    "recall_max": np.max(recalls),
                    "filter_rate_mean": np.mean(filter_rates),
                    "filter_rate_min": np.min(filter_rates),
                    "filter_rate_max": np.max(filter_rates),
                }
            )
    return pd.DataFrame(rows)


def plot_sim_filter_sensitivity() -> None:
    """Plot similarity filter recall and filter rate vs threshold."""
    sim_filter_df = load_sim_filter_results()

    claim_order = sorted(sim_filter_df["claim_id"].unique(), key=lambda c: int(c[1:]))

    _, (ax_recall, ax_filter_rate) = plt.subplots(1, 2, figsize=(5, 2), sharex=True)

    for claim_id in claim_order:
        claim_df = sim_filter_df[sim_filter_df["claim_id"] == claim_id].sort_values(
            "threshold"
        )
        thresholds = claim_df["threshold"]

        (line_recall,) = ax_recall.plot(
            thresholds, claim_df["recall_mean"], marker="o", label=claim_id
        )
        ax_recall.fill_between(
            thresholds,
            claim_df["recall_min"],
            claim_df["recall_max"],
            alpha=0.25,
            color=line_recall.get_color(),
        )

        (line_filter_rate,) = ax_filter_rate.plot(
            thresholds,
            claim_df["filter_rate_mean"],
            marker="o",
            label=claim_id,
        )
        ax_filter_rate.fill_between(
            thresholds,
            claim_df["filter_rate_min"],
            claim_df["filter_rate_max"],
            alpha=0.25,
            color=line_filter_rate.get_color(),
        )

    for ax in (ax_recall, ax_filter_rate):
        ax.locator_params(axis="both", nbins=6)
        ax.axvline(x=0.15, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Similarity Threshold")

    ax_recall.set_ylabel("Recall")
    ax_filter_rate.set_ylabel("Filter Rate")
    ax_filter_rate.legend(fontsize="xx-small")

    save_figure("sim_filter_sensitivity")
    sim_filter_df.to_csv(FIGURES_DIR / "sim_filter_sensitivity.csv", index=False)


def plot_ablation_per_claim(df: pd.DataFrame) -> None:
    """Save per-claim cost/latency/correctness for each ablation vs evg_opt."""
    ablation_df = df[
        (df["model"] == ABLATION_MODEL) | (df["implementation"] == "evg_ref")
    ]
    baseline_key = f"evg_opt_{ABLATION_MODEL}"
    reference = ablation_df[ablation_df["implementation"] == "evg_ref"].set_index(
        "claim"
    )["verification_result"]

    rows: list[dict[str, object]] = []
    for impl, label in _ABLATION_STEP_LABELS.items():
        if impl == "evg_opt":
            continue
        abl_key = f"{impl}_{ABLATION_MODEL}"
        claim_ids = ABLATION_APPLICABLE_CLAIMS[impl]
        for cid in sorted(claim_ids, key=lambda c: int(c[1:])):
            claim = CLAIM_ID_TO_CLAIM[cid]
            base = ablation_df[
                (ablation_df["impl_model"] == baseline_key)
                & (ablation_df["claim"] == claim)
            ]
            abl = ablation_df[
                (ablation_df["impl_model"] == abl_key) & (ablation_df["claim"] == claim)
            ]
            ref_val = reference.get(claim)
            opt_correct = (base["verification_result"] == ref_val).sum()
            abl_correct = (abl["verification_result"] == ref_val).sum()
            n_trials = len(base)

            for metric, col, decimals in [
                ("Cost", "total_token_cost", 3),
                ("Latency", "latency", 1),
            ]:
                rows.append(
                    {
                        "Implementation": label,
                        "Claim": cid,
                        "Metric": metric,
                        "OPT Mean": round(base[col].mean(), decimals),
                        "OPT Min": round(base[col].min(), decimals),
                        "OPT Max": round(base[col].max(), decimals),
                        "ABL Mean": round(abl[col].mean(), decimals),
                        "ABL Min": round(abl[col].min(), decimals),
                        "ABL Max": round(abl[col].max(), decimals),
                    }
                )
            rows.append(
                {
                    "Implementation": label,
                    "Claim": cid,
                    "Metric": "Correctness",
                    "OPT Mean": f"{opt_correct}/{n_trials}",
                    "ABL Mean": f"{abl_correct}/{n_trials}",
                }
            )

    pd.DataFrame(rows).to_csv(FIGURES_DIR / "ablation_per_claim.csv", index=False)


def plot_cost_f1_pareto(df: pd.DataFrame) -> None:
    """Scatter plot of mean F1 vs mean cost per claim, with Pareto frontier."""
    reference = df[df["implementation"] == "evg_ref"].set_index("claim")[
        "verification_result"
    ]

    # Aggregate F1 and cost per implementation–model pair across trials.
    records: list[dict[str, Any]] = []
    for impl_model in get_impl_models(df):
        impl_df = df[df["impl_model"] == impl_model]
        impl = impl_df["implementation"].iloc[0]
        model = impl_df["model"].iloc[0]

        trial_f1s: list[float] = []
        trial_costs: list[float] = []
        for trial_id in impl_df["trial_id"].unique():
            trial_df = impl_df[impl_df["trial_id"] == trial_id]
            trial_claims = trial_df.set_index("claim")["verification_result"]
            common = reference.index.intersection(trial_claims.index)
            ref, pred = reference.loc[common], trial_claims.loc[common]
            tp = (~ref & ~pred).sum()
            fp = (ref & ~pred).sum()
            fn = (~ref & pred).sum()
            _, _, f1 = compute_precision_recall_f1(tp, fp, fn)
            trial_f1s.append(f1)
            trial_costs.append(trial_df["total_token_cost"].mean())

        records.append(
            {
                "implementation": impl,
                "model": model,
                "f1": float(np.mean(trial_f1s)),
                "cost": float(np.mean(trial_costs)),
            }
        )

    points = pd.DataFrame(records)

    # Pareto frontier: point i is dominated if some j has f1 >= f1[i] and
    # cost <= cost[i] with at least one strict inequality.
    costs = points["cost"].values
    f1s = points["f1"].values
    is_pareto = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if (
                i != j
                and f1s[j] >= f1s[i]
                and costs[j] <= costs[i]
                and (f1s[j] > f1s[i] or costs[j] < costs[i])
            ):
                is_pareto[i] = False
                break

    _, ax = plt.subplots(figsize=(5, 2.5))

    markers = ["o", "s", "D", "^"]
    colors = ["C0", "C1", "C3", "C2"]  # make evg_opt green

    for impl, marker, color in zip(IMPLEMENTATIONS, markers, colors, strict=True):
        subset = points[points["implementation"] == impl]
        if subset.empty:
            continue
        ax.scatter(
            subset["cost"],
            subset["f1"],
            label=impl,
            marker=marker,
            color=color,
        )

    for _, row in points.iterrows():
        ax.annotate(
            _MODEL_LABELS[row["model"]],
            (row["cost"], row["f1"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize="x-small",
        )

    pareto_points = points[is_pareto].sort_values("cost")
    ax.plot(
        pareto_points["cost"],
        pareto_points["f1"],
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0.017, 19)
    ax.set_ylim(0.7, 1.04)
    ax.legend(fontsize="x-small")

    save_figure("cost_f1_pareto")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")

    df = load_results()
    print(f"Loaded {len(df)} results")
    print(df[["claim", "impl_model", "verification_result"]].to_string())

    print("\nGround Truth:")
    print(
        df[df["implementation"] == "evg_ref"][
            ["claim", "verification_result"]
        ].to_string(index=False)
    )

    # Summary plots (aggregated across claims)
    plot_verification_quality(df)
    plot_cost_f1_pareto(df)
    plot_provenance_precision_aggregated(df)
    plot_filter_quality_aggregated(df)
    plot_map_accuracy_aggregated(df)
    plot_mean_aggregated(
        df,
        "total_token_cost",
        "avg. token cost per claim ($)",
        "total_token_cost_aggregated",
        FormatType.COST,
    )
    plot_mean_aggregated(
        df,
        "latency",
        "avg. latency per claim (s)",
        "latency_aggregated",
        FormatType.LATENCY,
    )

    # Ablations
    plot_ablation_quality(df)
    plot_ablation_cost_and_latency(df)
    plot_ablation_per_claim(df)

    # Sensitivity analysis
    plot_sim_filter_sensitivity()

    # Per-claim plots
    plot_verification_heatmap(df)
    plot_metric(
        df,
        column="filter_precision",
        ylabel="filter precision",
        filename="filter_precision",
        format_type=FormatType.SCORE,
        impls=["evg_unopt", "evg_opt"],
    )
    plot_metric(
        df,
        column="filter_recall",
        ylabel="filter recall",
        filename="filter_recall",
        format_type=FormatType.SCORE,
        impls=["evg_unopt", "evg_opt"],
    )
    plot_metric(
        df,
        column="filter_f1",
        ylabel="filter f1 score",
        filename="filter_f1",
        format_type=FormatType.SCORE,
        impls=["evg_unopt", "evg_opt"],
    )
    plot_metric(
        df,
        column="map_accuracy",
        ylabel="map accuracy",
        filename="map_accuracy",
        format_type=FormatType.SCORE,
        impls=["evg_unopt", "evg_opt"],
    )
    plot_metric(
        df,
        column="provenance_precision",
        ylabel="provenance precision",
        filename="provenance_precision",
        format_type=FormatType.SCORE,
        impls=["evg_unopt", "evg_opt"],
    )
    plot_metric(
        df,
        column="lm_call_count",
        ylabel="LM call count",
        filename="lm_call_count",
        format_type=FormatType.COUNT,
        log_scale=True,
    )
    plot_metric(
        df,
        column="input_token_count",
        ylabel="input token count",
        filename="input_token_count",
        format_type=FormatType.TOKENS,
        log_scale=True,
    )
    plot_metric(
        df,
        column="output_token_count",
        ylabel="output token count",
        filename="output_token_count",
        format_type=FormatType.TOKENS,
        log_scale=True,
    )
    plot_metric(
        df,
        column="input_token_cost",
        ylabel="input token cost ($)",
        filename="input_token_cost",
        format_type=FormatType.COST,
        log_scale=True,
    )
    plot_metric(
        df,
        column="output_token_cost",
        ylabel="output token cost ($)",
        filename="output_token_cost",
        format_type=FormatType.COST,
        log_scale=True,
    )
    plot_metric(
        df,
        column="total_token_cost",
        ylabel="total token cost ($)",
        filename="total_token_cost",
        format_type=FormatType.COST,
        log_scale=True,
    )
    plot_metric(
        df,
        column="latency",
        ylabel="latency (s)",
        filename="latency",
        format_type=FormatType.LATENCY,
        log_scale=True,
    )

    print(f"\nFigures saved to {FIGURES_DIR}")

    if PAPER_DIR.is_dir():
        PAPER_FIGURES_DIR.mkdir(exist_ok=True)
        for pdf in FIGURES_DIR.glob("*.pdf"):
            if pdf.name in {
                "verification_result.pdf",
                "ablation_cost_and_latency.pdf",
                "sim_filter_sensitivity.pdf",
                "cost_f1_pareto.pdf",
            }:
                shutil.copy2(pdf, PAPER_FIGURES_DIR / pdf.name)
        print(f"Figures copied to {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
