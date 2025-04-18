"""
This script is used to evaluate a RAG system on the set of evaluation questions.

A cell-based script was preferred over a command-line script because it solved an issue
with the asynchronous evaluation of feedback functions:
https://github.com/truera/trulens/issues/915#issuecomment-1961889522
"""

# %%
import os
import sys
from pathlib import Path

import nest_asyncio
import yaml
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

project_root_path = Path(__file__).parents[1]
sys.path.append(str(project_root_path))

from src.mythesis_chatbot.evaluation import (  # NOQA E402
    get_prebuilt_trulens_recorder,
    run_evals,
)
from src.mythesis_chatbot.rag_setup import (  # NOQA E402
    SupportedRags,
    automerging_retrieval_setup,
    basic_rag_setup,
    sentence_window_retrieval_setup,
)

nest_asyncio.apply()
tru = TruSession(database_url=os.getenv("SUPABASE_DEV_CONNECTION_STRING_IPV4"))

# %%
run_dashboard(tru)

# %%

rag_mode: SupportedRags = "auto-merging retrieval"

input_file = project_root_path / "data/Master_Thesis.pdf"
save_dir = project_root_path / "data/indices/"
config_dir = project_root_path / "configs/"
questions_file = project_root_path / "data/eval_questions.txt"


match rag_mode:
    case "classic retrieval":
        with open(config_dir / "basic.yaml") as f:
            config = yaml.safe_load(f)
        engine = basic_rag_setup(input_file, save_dir, **config)

    case "auto-merging retrieval":
        with open(config_dir / "auto_merging.yaml") as f:
            config = yaml.safe_load(f)
        engine = automerging_retrieval_setup(input_file, save_dir, **config)

    case "sentence window retrieval":
        with open(config_dir / "sentence_window.yaml") as f:
            config = yaml.safe_load(f)
        engine = sentence_window_retrieval_setup(input_file, save_dir, **config)

tru_recorder = get_prebuilt_trulens_recorder(engine, config)

print("\n -- Running evaluation questions --\n")
run_evals(questions_file, tru_recorder, engine)
tru_recorder.wait_for_feedback_results()
