# %%
import os
import pandas as pd
import nest_asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mythesis_chatbot import evaluation
from trulens.core import TruSession
from mythesis_chatbot.rag_setup import (
    sentence_window_retrieval_setup,
)
import yaml
from trulens.dashboard.display import get_feedback_result
from trulens.dashboard import run_dashboard

# %%

with open(os.path.join("../../configs", "sentence_window.yaml"), "r") as f:
    config = yaml.safe_load(f)

engine = sentence_window_retrieval_setup(
    input_file="../../data/Master_Thesis.pdf", save_dir="../../data/indices", **config
)

# database_url=os.getenv("SUPABASE_CONNECTION_STRING")
tru = TruSession(database_url=os.getenv("SUPABASE_CONNECTION_STRING"))
tru.reset_database()
nest_asyncio.apply()


# %%

tru_recorder = evaluation.get_prebuilt_trulens_recorder(engine, config)
# %%

query = "Why?"
with tru_recorder as recording:  # noqa: F841
    response = engine.query(query)  # noqa: F841

# %%
database = tru_recorder.db

# %%

rec = recording.get()
# get_feedback_result(rec, "Context Relevance")

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)
    # database.insert_feedback(feedback_result)


# %%
evaluation.run_evals(
    os.path.join("../../data/", "eval_questions.txt"), tru_recorder, engine
)

# %%
records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()
# %%
pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]
# %%

tru.get_leaderboard(app_ids=[])
# %%
tru.run_dashboard()
