import os
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)


def get_openai_api_key():
    """
    Get the OpenAI API key from an environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "OpenAI API key not found. Please follow the instruction in the readme file."
    )


def build_sentence_window_index(
    documents,
    llm,
    embed_model,
    save_dir,
    sentence_window_size=3,
):
    """
    Parameters
    ----------
    documents : _type_
        _description_
    llm : _type_
        _description_
    embed_model : _type_
        _description_
    save_dir : str
        _description_
    sentence_window_size : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """
    # Create the sentence window node parser w/ default settings.
    # A node is a chunck of text. Each node returned by the sentence window node parser
    # also contains its context as metadata (closest chuncks of texts)
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.llm = llm
    Settings.node_parser = node_parser
    Settings.embed_model = embed_model

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(documents)
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # Used to replace the node content with a field from the node metadata.
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # Rerank can speed up an LLM query without sacrificing accuracy. It does so by
    # pruning away irrelevant nodes from the context.
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-MiniLM-L-2-v2"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine
