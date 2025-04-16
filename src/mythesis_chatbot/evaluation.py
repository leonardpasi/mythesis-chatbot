from pathlib import Path

import numpy as np
from tqdm import tqdm
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

from mythesis_chatbot.utils import get_config_hash


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
def f_answer_relevance(provider=OpenAI(), name="Answer Relevance"):
    return Feedback(provider.relevance_with_cot_reasons, name=name).on_input_output()


# Feedback function
def f_context_relevance(
    provider=OpenAI(),
    context=TruLlama.select_source_nodes().node.text,
    name="Context Relevance",
):
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
):
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
):
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
