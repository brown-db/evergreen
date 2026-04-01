import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import nltk  # type: ignore
from nltk.tokenize import sent_tokenize  # type: ignore
from pydantic import BaseModel

from evergreen.common.constants import JSON_INDENT
from evergreen.model.language_model import LanguageModel

try:
    nltk.data.find("tokenizers/punkt_tab")  # type: ignore
except LookupError:
    nltk.download("punkt_tab", quiet=True)  # type: ignore

logger = logging.getLogger(__name__)

# The general structure of the below prompt is adapted from [1].
# The specific few shot examples are taken from [2], which was also done in [1].
# References:
# 1. Wei et al. (2024). "Long-form factuality in large language models".
#    Table 11, p. 29.
#    https://proceedings.neurips.cc/paper_files/paper/2024/file/937ae0e83eb08d2cb8627fe1def8c751-Paper-Conference.pdf
# 2. Min et al. (2023). "FACTSCORE: Fine-grained Atomic Evaluation of Factual Precision
#    in Long Form Text Generation".
#    Table 15, p. 12098
#    https://aclanthology.org/2023.emnlp-main.741.pdf
_SPLIT_PROMPT_TEMPLATE = """<instructions>
1. You are given a sentence.
   Your task is to break the sentence down into a list of atomic facts.
2. An atomic fact is a sentence containing a singular piece of information.
3. Each atomic fact in the outputted list should check a different piece of information.
4. Use the examples below to learn how to do this.
5. Your task is to process the text inside the <sentence> tags at the bottom.
</instructions>

<examples>
{examples}
</examples>

<sentence>
{sentence}
</sentence>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>"""

_SPLIT_EXAMPLES: list[dict[str, object]] = [
    {
        "sentence": "He made his acting debut in the film The Moon is the Sun's Dream "
        "(1992), and continued to appear in small and supporting roles throughout the "
        "1990s.",
        "facts": [
            "He made his acting debut in the film.",
            "He made his acting debut in The Moon is the Sun's Dream.",
            "The Moon is the Sun's Dream is a film.",
            "The Moon is the Sun's Dream was released in 1992.",
            "After his acting debut, he appeared in small and supporting roles.",
            "After his acting debut, he appeared in small and supporting roles "
            "throughout the 1990s.",
        ],
    },
    {
        "sentence": "He is also a successful producer and engineer, having worked with "
        "a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor "
        "Swift.",
        "facts": [
            "He is successful.",
            "He is a producer.",
            "He is a engineer.",
            "He has worked with a wide variety of artists.",
            "Willie Nelson is an artist.",
            "He has worked with Willie Nelson.",
            "Tim McGraw is an artist.",
            "He has worked with Tim McGraw.",
            "Taylor Swift is an artist.",
            "He has worked with Taylor Swift.",
        ],
    },
    {
        "sentence": "In 1963, Collins became one of the third group of astronauts "
        "selected by NASA and he served as the back-up Command Module Pilot for the "
        "Gemini 7 mission.",
        "facts": [
            "Collins became an astronaut.",
            "Collins became one of the third group of astronauts.",
            "Collins became one of the third group of astronauts selected.",
            "Collins became one of the third group of astronauts selected by NASA.",
            "Collins became one of the third group of astronauts selected by NASA in "
            "1963.",
            "He served as the Command Module Pilot.",
            "He served as the back-up Command Module Pilot.",
            "He served as the Command Module Pilot for the Gemini 7 mission.",
        ],
    },
    {
        "sentence": "In addition to his acting roles, Bateman has written and directed "
        "two short films and is currently in development on his feature debut.",
        "facts": [
            "Bateman has acting roles.",
            "Bateman has written two short films.",
            "Bateman has directed two short films.",
            "Bateman has written and directed two short films.",
            "Bateman is currently in development on his feature debut.",
        ],
    },
    {
        "sentence": "Michael Collins (born October 31, 1930) is a retired American "
        "astronaut and test pilot who was the Command Module Pilot for the Apollo 11 "
        "mission in 1969.",
        "facts": [
            "Michael Collins was born on October 31, 1930.",
            "Michael Collins is retired.",
            "Michael Collins is an American.",
            "Michael Collins was an astronaut.",
            "Michael Collins was a test pilot.",
            "Michael Collins was the Command Module Pilot.",
            "Michael Collins was the Command Module Pilot for the Apollo 11 mission.",
            "Michael Collins was the Command Module Pilot for the Apollo 11 mission in "
            "1969.",
        ],
    },
    {
        "sentence": "He was an American composer, conductor, and musical director.",
        "facts": [
            "He was an American.",
            "He was a composer.",
            "He was a conductor.",
            "He was a musical director.",
        ],
    },
    {
        "sentence": "She currently stars in the romantic comedy series, Love and "
        "Destiny, which premiered in 2019.",
        "facts": [
            "She currently stars in Love and Destiny.",
            "Love and Destiny is a romantic comedy series.",
            "Love and Destiny premiered in 2019.",
        ],
    },
    {
        "sentence": "During his professional career, McCoy played for the Broncos, the "
        "San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars.",
        "facts": [
            "McCoy played for the Broncos.",
            "McCoy played for the Broncos during his professional career.",
            "McCoy played for the San Diego Chargers.",
            "McCoy played for the San Diego Chargers during his professional career.",
            "McCoy played for the Minnesota Vikings.",
            "McCoy played for the Minnesota Vikings during his professional career.",
            "McCoy played for the Jacksonville Jaguars.",
            "McCoy played for the Jacksonville Jaguars during his professional career.",
        ],
    },
]

# The below prompt is adapted from [1].
# References:
# 1. Wei et al. (2024). "Long-form factuality in large language models".
#    Table 12, p. 31.
#    https://proceedings.neurips.cc/paper_files/paper/2024/file/937ae0e83eb08d2cb8627fe1def8c751-Paper-Conference.pdf
_REVISE_FACT_PROMPT_TEMPLATE = """<instructions>
Vague references include but are not limited to:
- Pronouns (e.g., “his”, “they”, “her”)
- Unknown entities (e.g., “this event”, “the research”, “the invention”)
- Non-full names (e.g., “Jeff...” or “Bezos...” when referring to Jeff Bezos)

1. The following <statement> has been extracted from the broader context of the
   given <response>.
2. Modify the <statement> by replacing vague references with the proper entities
   from the <response> that they are referring to.
3. You MUST NOT change any of the factual claims made by the original <statement>.
4. You MUST NOT add any additional factual claims to the original <statement>.
   For example, given the response "Titanic is a movie starring Leonardo
   DiCaprio," the <statement> "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your reasoning.
   As part of your reasoning, be sure to identify the subjects in the
   <statement> and determine whether they are vague references.
   If they are vague references, identify the proper entity that they are
   referring to and be sure to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement.
7. Your task is to do this for the last <statement> and <response>.
   Some examples have been provided for you to learn how to do this task.
</instructions>

<examples>
{examples}
</examples>

<statement>
{statement}
</statement>

<response>
{response}
</response>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>"""

_REVISE_FACT_EXAMPLES: list[dict[str, object]] = [
    {
        "statement": "Acorns is a company.",
        "response": "Acorns is a financial technology company founded in 2012 by "
        "Walter Cruttenden, Jeff Cruttenden, and Mark Dru that provides "
        "micro-investing services. The company is headquartered in Irvine, California.",
        "output": {
            "reasoning": 'The subject in the statement "Acorns is a company" is '
            '"Acorns". "Acorns" is not a pronoun and does not reference an unknown '
            'entity. Furthermore, "Acorns" is not further specified in the response, '
            'so we can assume that it is a full name. Therefore "Acorns" is not a '
            "vague reference.",
            "revised_statement": "Acorns is a company.",
        },
    },
    {
        "statement": "He teaches courses on deep learning.",
        "response": "After completing his Ph.D., Quoc Le joined Google Brain, where he "
        "has been working on a variety of deep learning projects. Le is also an "
        "adjunct professor at the University of Montreal, where he teaches courses on "
        "deep learning.",
        "output": {
            "reasoning": 'The subject in the statement "He teaches course on deep '
            'learning" is "he". From the response, we can see that this statement '
            'comes from the sentence "Le is also an adjunct professor at the '
            'University of Montreal, where he teaches courses on deep learning.", '
            'meaning that "he" refers to "Le". From the response, we can also see '
            'that "Le" refers to "Quoc Le". Therefore "Le" is a non-full name that '
            'should be replaced by "Quoc Le."',
            "revised_statement": "Quoc Le teaches courses on deep learning.",
        },
    },
    {
        "statement": 'The television series is called "You\'re the Worst."',
        "response": "Xochitl Gomez began her acting career in theater productions, and "
        "she made her television debut in 2016 with a guest appearance on the Disney "
        'Channel series "Raven\'s Home." She has also appeared in the television '
        'series "You\'re the Worst" and "Gentefied."',
        "output": {
            "reasoning": 'The subject of the statement "The television series is '
            'called "You\'re the Worst."" is "the television series". This is a '
            "reference to an unknown entity, since it is unclear what television "
            'series is "the television series". From the response, we can see that the '
            "statement is referring to the television series that Xochitl Gomez "
            'appeared in. Thus, "the television series" is a vague reference that '
            'should be replaced by "the television series that Xochitl Gomez appeared '
            'in".',
            "revised_statement": "The television series that Xochitl Gomez appeared in "
            'is called "You\'re the Worst.".',
        },
    },
    {
        "statement": "Dean joined Google.",
        "response": "Jeff Dean is a Google Senior Fellow and the head of Google AI, "
        "leading research and development in artificial intelligence. Dean joined "
        "Google in 1999 and has been essential to its continued development in the "
        "field.",
        "output": {
            "reasoning": 'The subject of the statement "Dean joined Google" is "Dean". '
            'From the response, we can see that "Dean" is the last name of '
            '"Jeff Dean". Therefore "Dean" is a non-full name, making it a vague '
            'reference. It should be replaced by "Jeff Dean", which is the full name.',
            "revised_statement": "Jeff Dean joined Google.",
        },
    },
]


class Facts(BaseModel):
    facts: list[str]


_FACTS_JSON_SCHEMA = Facts.model_json_schema()
_FACTS_JSON_SCHEMA_STR = json.dumps(_FACTS_JSON_SCHEMA, indent=JSON_INDENT)


class RevisedStatement(BaseModel):
    reasoning: str
    revised_statement: str


_REVISED_STATEMENT_JSON_SCHEMA = RevisedStatement.model_json_schema()
_REVISED_STATEMENT_JSON_SCHEMA_STR = json.dumps(
    _REVISED_STATEMENT_JSON_SCHEMA, indent=JSON_INDENT
)


class ClaimDecomposer:
    _MAX_WORKERS = 32

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def decompose(self, text: str) -> list[str]:
        sentences = sent_tokenize(text)

        logger.debug(
            "Decomposing text into claims: sentence_count=%d, sentences=%s",
            len(sentences),
            sentences,
        )

        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as executor:
            nested_facts = list(executor.map(self._split, sentences))

        all_facts = [fact for facts in nested_facts for fact in facts]

        revise_with_context = partial(self._revise, aggregate_response=text)

        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as executor:
            claims = list(executor.map(revise_with_context, all_facts))

        logger.debug("Decomposition complete: %d claims", len(claims))

        return claims

    def _split(self, sentence: str) -> list[str]:
        examples = [
            f"<example_{i}>\n"
            "<sentence>\n"
            f"{example['sentence']}\n"
            "</sentence>\n"
            "<facts>\n"
            f"{json.dumps({'facts': example['facts']}, indent=JSON_INDENT)}\n"
            "</facts>\n"
            f"</example_{i}>"
            for i, example in enumerate(_SPLIT_EXAMPLES, start=1)
        ]

        prompt = _SPLIT_PROMPT_TEMPLATE.format(
            examples="\n".join(examples),
            sentence=sentence,
            json_schema=_FACTS_JSON_SCHEMA_STR,
        )

        response = self._language_model.prompt_with_schema(
            prompt, Facts, _FACTS_JSON_SCHEMA
        )

        facts = response.facts

        return facts

    def _revise(self, fact: str, aggregate_response: str) -> str:
        examples = [
            f"<example_{i}>\n"
            "<statement>\n"
            f"{example['statement']}\n"
            "</statement>\n"
            "<response>\n"
            f"{example['response']}\n"
            "</response>\n"
            "<output>\n"
            f"{json.dumps(example['output'], indent=JSON_INDENT)}\n"
            "</output>\n"
            f"</example_{i}>"
            for i, example in enumerate(_REVISE_FACT_EXAMPLES, start=1)
        ]

        prompt = _REVISE_FACT_PROMPT_TEMPLATE.format(
            examples="\n".join(examples),
            statement=fact,
            response=aggregate_response,
            json_schema=_REVISED_STATEMENT_JSON_SCHEMA_STR,
        )
        response = self._language_model.prompt_with_schema(
            prompt, RevisedStatement, _REVISED_STATEMENT_JSON_SCHEMA
        )

        revised_fact = response.revised_statement

        return revised_fact
