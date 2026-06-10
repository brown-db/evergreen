from evergreen.catalog.schema import Field, Schema
from evergreen.claim_compiler import ClaimCompiler
from evergreen.core.session_context import SessionContext
from evergreen.model.config import CortexModelConfig
from evergreen.storage.row import Row
from tests.conftest import DEFAULT_CONNECTION_NAME, DEFAULT_LANGUAGE_MODEL


def test_claim_compiler():
    model_config = CortexModelConfig(
        language_models=[DEFAULT_LANGUAGE_MODEL],
        embedding_model="",
        connection_name=DEFAULT_CONNECTION_NAME,
    )
    language_model = model_config.create_language_model(cache_dir=None)
    compiler = ClaimCompiler(language_model)

    agg_prompt = "Summarize the reviews for this restaurant."
    schema = Schema(
        (
            Field("id", int, "The ID of the review"),
            Field("review", str, "The text of the review"),
        ),
        ("id",),
    )
    claim = "Some customers wait 20+ minutes to be acknowledged."
    result = compiler.compile(agg_prompt, schema, claim, hints="")

    assert len(result.response) > 0
    assert "check(" in result.query
    assert len(result.language_model_metrics) == 1
    assert result.language_model_metrics[0].prompt_count == 1
    assert result.latency > 0


def test_build_query(default_ctx: SessionContext):
    schema = Schema(
        (Field("id", int), Field("company_id", str), Field("is_complaint", bool)),
        ("id",),
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "A", False)),
        Row((3, "B", True)),
        Row((4, "B", True)),
    ]
    df = default_ctx.read_rows(rows, schema)

    query_src = """(
    df.aggregate(
        [proportion(col("is_complaint")).alias("complaint_prop")],
        group_by=[col("company_id")]
    )
    .aggregate(
        [count_if(col("complaint_prop") > 0.5).alias("num_companies")]
    )
    .check(col("num_companies").eq(1))
)"""

    result = ClaimCompiler.build_query(df, query_src).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, True))]


def test_build_query_with_enum(default_ctx: SessionContext):
    schema = Schema(
        (Field("id", int), Field("company_id", str), Field("sentiment", str)),
        ("id",),
    )
    rows = [
        Row((1, "A", "mixed")),
        Row((2, "A", "positive")),
        Row((3, "A", "negative")),
        Row((4, "B", "positive")),
        Row((5, "B", "positive")),
    ]
    df = default_ctx.read_rows(rows, schema)

    query_src = """class Satisfaction(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"

(
    df.aggregate(
        [
            proportion(
                col("sentiment").eq(Satisfaction.POSITIVE.value)
                | col("sentiment").eq(Satisfaction.MIXED.value)
            ).alias("positive_or_mixed_prop"),
            proportion(
                col("sentiment").eq(Satisfaction.NEGATIVE.value)
                | col("sentiment").eq(Satisfaction.MIXED.value)
            ).alias("negative_or_mixed_prop"),
        ],
        group_by=[col("company_id")]
    )
    .aggregate(
        [
            bool_or(
                (col("positive_or_mixed_prop") >= 0.2)
                & (col("negative_or_mixed_prop") >= 0.2)
            ).alias("some_mixed")
        ]
    )
    .check(col("some_mixed"))
)"""

    result = ClaimCompiler.build_query(df, query_src).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, True))]
