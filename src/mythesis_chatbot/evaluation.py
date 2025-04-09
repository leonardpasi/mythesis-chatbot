from pathlib import Path

import numpy as np
from tqdm import tqdm
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
from trulens.providers.openai import OpenAI


def run_evals(eval_questions_path: Path, tru_recorder, query_engine):

    eval_questions = []
    with open(eval_questions_path, "r") as file:
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


def get_prebuilt_trulens_recorder(query_engine, app_id: str):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=[f_answer_relevance(), f_context_relevance(), f_groundedness()],
    )
    return tru_recorder
