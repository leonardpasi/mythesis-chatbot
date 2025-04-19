import os
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback, TruSession
from trulens.providers.openai import OpenAI

from src.mythesis_chatbot.utils import get_config_hash


def run_evals(eval_questions_path: Path, tru_recorder, query_engine):

    eval_questions = []
    with open(eval_questions_path) as file:
        for line in file:
            item = line.strip()
            eval_questions.append(item)

    for question in tqdm(eval_questions):
        with tru_recorder as recording:  # noqa: F841
            response = query_engine.query(question)  # noqa: F841


# Feedback function
def f_answer_relevance(provider=OpenAI(), name="Answer Relevance") -> Feedback:
    return Feedback(provider.relevance_with_cot_reasons, name=name).on_input_output()


# Feedback function
def f_context_relevance(
    provider=OpenAI(),
    context=TruLlama.select_source_nodes().node.text,
    name="Context Relevance",
) -> Feedback:
    return (
        Feedback(provider.relevance, name=name)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )


# Feedback function
def f_groundedness(
    provider=OpenAI(),
    context=TruLlama.select_source_nodes().node.text,
    name="Groundedness",
) -> Feedback:
    return (
        Feedback(
            provider.groundedness_measure_with_cot_reasons,
            name=name,
        )
        .on(context)
        .on_output()
    )


def get_prebuilt_trulens_recorder(
    query_engine, query_engine_config: dict[str, str | int]
) -> TruLlama:
    app_name = query_engine_config["rag_mode"]
    app_version = get_config_hash(query_engine_config)

    tru_recorder = TruLlama(
        query_engine,
        app_name=app_name,
        app_version=app_version,
        metadata=query_engine_config,
        feedbacks=[f_answer_relevance(), f_context_relevance(), f_groundedness()],
    )
    return tru_recorder


def get_tru_session(database: Literal["prod", "dev"]) -> TruSession:

    print(f"Connecting to {database.lower()} database...")

    match database.lower():
        case "prod":
            database_url = os.getenv("SUPABASE_PROD_CONNECTION_STRING_IPV4")
            if database_url is None:
                raise RuntimeError(
                    "IPv4 connection string to production database is not available as"
                    " an environment variable."
                )
            else:
                print("Using IPv4 connection string...")
                tru = TruSession(database_url=database_url)
                return tru

        case "dev":
            database_url = os.getenv("SUPABASE_DEV_CONNECTION_STRING_IPV6")
            if database_url:
                try:
                    print("Using IPv6 connection string...")
                    tru = TruSession(database_url=database_url)
                    return tru
                except Exception as e:
                    print(
                        "An error occurred while connecting to remote dev database with"
                        f" IPv6 connection string: {e}"
                    )
                    print("Reverting to IPv4")
            else:
                print(
                    "IPv6 connection string to dev database is not available as an"
                    " environment variable. Reverting to IPv4."
                )

            database_url = os.getenv("SUPABASE_DEV_CONNECTION_STRING_IPV4")
            if database_url is None:
                raise RuntimeError(
                    "IPv4 connection string to dev database is not available"
                    " as an environment variable."
                )
            else:
                tru = TruSession(database_url=database_url)
                return tru
        case _:
            raise ValueError(
                f"Invalid database: {database}. Choose betwen 'prod' and 'dev'"
            )
