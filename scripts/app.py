import os
from pathlib import Path

import gradio as gr
import nest_asyncio
import yaml
from trulens.core import TruSession

from src.mythesis_chatbot.evaluation import get_prebuilt_trulens_recorder
from src.mythesis_chatbot.rag_setup import (
    SupportedRags,
    automerging_retrieval_setup,
    basic_rag_setup,
    sentence_window_retrieval_setup,
)

input_file_dir = Path(__file__).parents[1] / "data/"
save_dir = Path(__file__).parents[1] / "data/indices/"
config_dir = Path(__file__).parents[1] / "configs/"
welcome_message_path = Path(__file__).parents[1] / "spaces/welcome_message.md"

# Enables running async code inside an existing event loop without crashing.
nest_asyncio.apply()

tru = TruSession(database_url=os.getenv("SUPABASE_PROD_CONNECTION_STRING_IPV4"))


class ChatBot:
    def __init__(
        self,
        input_file_dir,
        save_dir,
        config_dir,
    ):
        self.recorder = None
        self.previous_rag_mode = None
        self.recorder = None

        with open(os.path.join(config_dir, "basic.yaml")) as f:
            self.basic_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, "auto_merging.yaml")) as f:
            self.automerging_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, "sentence_window.yaml")) as f:
            self.sentence_window_config = yaml.safe_load(f)

        self.basic_engine = basic_rag_setup(
            input_file=os.path.join(input_file_dir, self.basic_config["source_doc"]),
            save_dir=save_dir,
            **self.basic_config,
        )
        self.automerging_engine = automerging_retrieval_setup(
            input_file=os.path.join(
                input_file_dir, self.automerging_config["source_doc"]
            ),
            save_dir=save_dir,
            **self.automerging_config,
        )
        self.sentence_window_engine = sentence_window_retrieval_setup(
            input_file=os.path.join(
                input_file_dir, self.sentence_window_config["source_doc"]
            ),
            save_dir=save_dir,
            **self.sentence_window_config,
        )

    def __call__(self, query: str, rag_mode: SupportedRags):

        match rag_mode:
            case "classic retrieval":

                if self.previous_rag_mode != rag_mode:
                    self.previous_rag_mode = rag_mode
                    self.recorder = get_prebuilt_trulens_recorder(
                        self.basic_engine, self.basic_config
                    )

                with self.recorder as recording:  # noqa: F841
                    response = self.basic_engine.query(query)

            case "auto-merging retrieval":
                if self.previous_rag_mode != rag_mode:
                    self.previous_rag_mode = rag_mode
                    self.recorder = get_prebuilt_trulens_recorder(
                        self.automerging_engine, self.automerging_config
                    )

                with self.recorder as recording:  # noqa: F841
                    response = self.automerging_engine.query(query)

            case "sentence window retrieval":
                if self.previous_rag_mode != rag_mode:
                    self.previous_rag_mode = rag_mode
                    self.recorder = get_prebuilt_trulens_recorder(
                        self.sentence_window_engine, self.sentence_window_config
                    )

                with self.recorder as recording:  # noqa: F841
                    response = self.sentence_window_engine.query(query)

        return response.response


chat_bot = ChatBot(input_file_dir, save_dir, config_dir)
default_message = (
    "Ask about a topic that is discussed in my master thesis."
    " E.g., what is this master thesis about? Or what is epistemic uncertainty?"
)

with open(welcome_message_path, encoding="utf-8") as f:
    description = f.read()

gradio_app = gr.Interface(
    fn=chat_bot,
    inputs=[
        gr.Textbox(placeholder=default_message, label="Query"),
        gr.Dropdown(
            choices=SupportedRags.__args__,
            label="RAG mode",
            value=SupportedRags.__args__[0],
        ),
    ],
    outputs=[
        gr.Textbox(label="Answer"),
    ],
    title="RAG powered chatbot",
    description=description,
)

gradio_app.launch()
