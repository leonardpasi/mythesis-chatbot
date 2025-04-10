import gradio as gr

from mythesis_chatbot.rag_setup import (
    SupportedRags,
    automerging_retrieval_setup,
    basic_rag_setup,
    sentence_window_retrieval_setup,
)

input_file = "./data/Master_Thesis.pdf"
save_dir = "./data/indices/"

automerging_engine = automerging_retrieval_setup(
    input_file=input_file,
    save_dir=save_dir,
    llm_openai_model="gpt-4o-mini",
    embed_model="BAAI/bge-small-en-v1.5",
    chunk_sizes=[2048, 512, 128],
    similarity_top_k=6,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n=2,
)

sentence_window_engine = sentence_window_retrieval_setup(
    input_file=input_file,
    save_dir=save_dir,
    llm_openai_model="gpt-4o-mini",
    embed_model="BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    similarity_top_k=6,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n=2,
)

basic_engine = basic_rag_setup(
    input_file=input_file,
    save_dir=save_dir,
    llm_openai_model="gpt-4o-mini",
    embed_model="BAAI/bge-small-en-v1.5",
    similarity_top_k=6,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n=2,
)


def chat_bot(query: str, rag_mode: SupportedRags) -> str:
    if rag_mode == "basic":
        return basic_engine.query(query).response
    if rag_mode == "auto-merging retrieval":
        return automerging_engine.query(query).response
    if rag_mode == "sentence window retrieval":
        return sentence_window_engine.query(query).response


default_message = (
    "Ask a about a topic that is discussed in my master thesis."
    "E.g., what is epistemic uncertainty?"
)

gradio_app = gr.Interface(
    fn=chat_bot,
    inputs=[
        gr.Textbox(placeholder=default_message),
        gr.Dropdown(
            choices=["basic", "sentence window retrieval", "auto-merging retrieval"],
            label="RAG mode",
            value="basic",
        ),
    ],
    outputs=["text"],
)

if __name__ == "__main__":
    gradio_app.launch()
