The below is documentation specifically for an LLM to consume.

::: evergreen.col
::: evergreen.lit
::: evergreen.prompt
::: evergreen.count_if
::: evergreen.proportion
::: evergreen.bool_and
::: evergreen.bool_or

::: evergreen.Expr
    options:
        members:
            - alias
            - eq
            - ne
            - __lt__
            - __le__
            - __gt__
            - __ge__
            - __and__
            - __or__
            - __invert__

::: evergreen.DataFrame
    options:
        members:
            - filter
            - map
            - aggregate
            - with_rank
            - check

## Query Examples

Check if some movie reviews mention beautiful cinematography:

```
>>> (
...     df.map(
...         prompt(
...             "Identify whether the {review} mentions beautiful cinematography", bool
...         ).alias("mentions_beautiful_cinematography")
...     )
...     .aggregate(
...         [bool_or(col("mentions_beautiful_cinematography")).alias("some_beautiful_cinematography")]
...     )
...     .check(col("some_beautiful_cinematography"))
... )
```

Check if all movies have majority positive reviews:

```
>>> (
...     df.map(
...         prompt(
...             "Identify whether the {review} has a positive sentiment towards the movie", bool
...         ).alias("is_positive")
...     )
...     .aggregate(
...         [proportion(col("is_positive")).alias("positive_prop")],
...         group_by=[col("movie_name")]
...     )
...     .aggregate(
...         [bool_and(col("positive_prop") > 0.5).alias("all_majority_positive")]
...     )
...     .check(col("all_majority_positive"))
... )
```

Check if excessive violence is a frequent complaint for multiple movies:

```
>>> (
...     df.filter(
...         prompt("The {review} mentions a complaint towards the movie")
...     )
...     .map(
...         prompt(
...             "Identify whether the {review}'s complaint is about excessive violence", bool
...         ).alias("about_violence")
...     )
...     .aggregate(
...         [proportion(col("about_violence")).alias("violence_prop")],
...         group_by=[col("movie_name")]
...     )
...     .aggregate(
...         [count_if(col("violence_prop") >= 0.3).alias("num_movies_with_freq_complaint")]
...     )
...     .check(col("num_movies_with_freq_complaint") >= 2)
... )
```

Check if 10% of reviews praise the movie's sound effects:

```
>>> (
...     df.map(
...         prompt(
...             "Identify whether the {review} praises the movie's sound effects", bool
...         ).alias("praises_sound_effects")
...     )
...     .aggregate(
...         [proportion(col("praises_sound_effects")).alias("praises_sound_effects_prop")]
...     )
...     .check(col("praises_sound_effects_prop").eq(0.10))
... )
```

Check if Bob Smith wrote the soundtrack for the movie:

```
>>> (
...     df.filter(
...         prompt("The {review} mentions the writer of the movie's soundtrack")
...     )
...     .map(
...         prompt(
...             "Identify whether the {review} says that the writer of the movie's soundtrack is Bob Smith", bool
...         ).alias("is_bob_smith")
...     )
...     .aggregate(
...         [bool_and(col("is_bob_smith")).alias("all_bob_smith")]
...     )
...     .check(col("all_bob_smith"))
... )
```

Check if some movies received mixed reviews:

```
>>> class Sentiment(Enum):
...     POSITIVE = "positive"
...     NEGATIVE = "negative"
...     MIXED = "mixed"
...     NEUTRAL = "neutral"
>>> (
...     df.map(
...         prompt(
...             "Identify the sentiment of the {review} as positive (mostly favorable), "
...             "negative (mostly unfavorable), mixed (both favorable and unfavorable), "
...             "or neutral (neither)", Sentiment
...         ).alias("sentiment")
...     )
...     .aggregate(
...         [
...             proportion(col("sentiment").eq(Sentiment.POSITIVE) | col("sentiment").eq(Sentiment.MIXED)).alias("positive_or_mixed_prop"),
...             proportion(col("sentiment").eq(Sentiment.NEGATIVE) | col("sentiment").eq(Sentiment.MIXED)).alias("negative_or_mixed_prop")
...         ],
...         group_by=[col("movie_name")]
...     )
...     .aggregate([bool_or((col("positive_or_mixed_prop") >= 0.2) & (col("negative_or_mixed_prop") >= 0.2)).alias("some_mixed")])
...     .check(col("some_mixed"))
... )
```

Check if Interstellar has the most highly rated soundtrack among all the movie reviews:

```
>>> (
...     df.filter(
...         prompt("The {review} mentions the movie's soundtrack")
...     )
...     .map(
...         prompt(
...             "Identify whether the {review} praises the movie's soundtrack", bool
...         ).alias("praises_soundtrack")
...     )
...     .aggregate(
...         [proportion(col("praises_soundtrack")).alias("praises_prop")],
...         group_by=[col("movie_name")]
...     )
...     .with_rank(col("praises_prop"))
...     .filter(col("movie_name").eq("Interstellar"))
...     .check(col("rank").eq(1))
... )
```

Check if Coco has the second highest number of praises among all the movie reviews:

```
>>> (
...     df.map(
...         prompt(
...             "Identify whether the {review} praises the movie", bool
...         ).alias("praises_movie")
...     )
...     .aggregate(
...         [count_if(col("praises_movie")).alias("praises_count")],
...         group_by=[col("movie_name")]
...     )
...     .with_rank(col("praises_count"))
...     .filter(col("movie_name").eq("Coco"))
...     .check(col("rank").eq(2))
... )
```

Check if none of the movie reviews complain about the visual effects
(using negation normal form):

```
>>> (
...     df.map(
...         prompt(
...             "Identify whether the {review} complains about the visual effects", bool
...         ).alias("complains_about_vfx")
...     )
...     .aggregate([bool_and(~col("complains_about_vfx")).alias("all_no_complaints")])
...     .check(col("all_no_complaints"))
... )
```
