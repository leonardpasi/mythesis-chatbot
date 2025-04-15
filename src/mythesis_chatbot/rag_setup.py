import json
import os
from typing import Literal

import openai
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from mythesis_chatbot.utils import get_config_hash, get_openai_api_key

SupportedRags = Literal[
    "classic retrieval", "sentence window retrieval", "auto-merging retrieval"
]
SupportedOpenAIllms = Literal["gpt-4o-mini", "gpt-3.5-turbo"]
SupportedEmbedModels = Literal["BAAI/bge-small-en-v1.5"]
SupportedRerankModels = Literal["cross-encoder/ms-marco-MiniLM-L-2-v2"]


def load_data(input_file: str) -> Document:

    reader = SimpleDirectoryReader(input_files=[input_file])
    documents = reader.load_data()  # List of Document objects (one object per page)
    # Merge into single document
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    return document


def build_sentence_window_index(
    input_file: str,
    save_dir: str,
    index_config: dict[str, str | int],
):
    config_hash = get_config_hash(index_config)
    save_dir = os.path.join(save_dir, "sentence_window", config_hash)

    Settings.embed_model = HuggingFaceEmbedding(model_name=index_config["embed_model"])

    if not os.path.exists(save_dir):

        document = load_data(input_file)

        # Create the sentence window node parser w/ default settings.
        # A node is a chunck of text. Each node returned by the sentence window node
        # parser also contains its context as metadata (closest chuncks of texts)
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=index_config["sentence_window_size"],
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        Settings.node_parser = node_parser

        sentence_index = VectorStoreIndex.from_documents([document])
        sentence_index.storage_context.persist(persist_dir=save_dir)
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(index_config, f, indent=2)

    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return sentence_index


def build_automerging_index(
    input_file: str,
    save_dir: str,
    index_config: dict[str, str | list[int]],
):

    config_hash = get_config_hash(index_config)
    save_dir = os.path.join(save_dir, "auto_merging", config_hash)

    Settings.embed_model = HuggingFaceEmbedding(model_name=index_config["embed_model"])

    if not os.path.exists(save_dir):

        document = load_data(input_file)
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=index_config["chunk_sizes"]
        )
        nodes = node_parser.get_nodes_from_documents([document])
        leaf_nodes = get_leaf_nodes(nodes)

        Settings.node_parser = node_parser

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(index_config, f, indent=2)

    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )
    return automerging_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
):
    # Used to replace the node content with a field from the node metadata.
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # Rerank can speed up an LLM query without sacrificing accuracy. It does so by
    # pruning away irrelevant nodes from the context.
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model)

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k: int = 12,
    rerank_top_n: int = 6,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model)
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine


def sentence_window_retrieval_setup(
    input_file: str,
    save_dir: str,
    llm_openai_model: SupportedOpenAIllms = "gpt-4o-mini",
    temperature: float = 0.1,
    embed_model: SupportedEmbedModels = "BAAI/bge-small-en-v1.5",
    sentence_window_size: int = 3,
    similarity_top_k: int = 6,
    rerank_model: SupportedRerankModels = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n: int = 2,
    **kwargs
):

    openai.api_key = get_openai_api_key()

    # This allows to uniquely identify the index
    config = {
        "doc_source": os.path.basename(input_file),
        "embed_model": embed_model,
        "sentence_window_size": sentence_window_size,
    }

    # 1. Build index
    index = build_sentence_window_index(input_file, save_dir, config)

    Settings.llm = OpenAI(model=llm_openai_model, temperature=temperature)

    # 2. Get engine
    sentence_window_engine = get_sentence_window_query_engine(
        index,
        similarity_top_k=similarity_top_k,
        rerank_model=rerank_model,
        rerank_top_n=rerank_top_n,
    )

    return sentence_window_engine


def automerging_retrieval_setup(
    input_file: str,
    save_dir: str,
    llm_openai_model: SupportedOpenAIllms = "gpt-4o-mini",
    temperature: float = 0.1,
    embed_model: SupportedEmbedModels = "BAAI/bge-small-en-v1.5",
    chunk_sizes=[2048, 512, 128],
    similarity_top_k: int = 6,
    rerank_model: SupportedRerankModels = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n: int = 2,
    **kwargs
):
    openai.api_key = get_openai_api_key()

    # This allows to uniquely identify the index
    config = {
        "doc_source": os.path.basename(input_file),
        "embed_model": embed_model,
        "chunk_sizes": chunk_sizes,
    }

    # 1. Build index
    index = build_automerging_index(input_file, save_dir, config)

    Settings.llm = OpenAI(model=llm_openai_model, temperature=temperature)

    # 2. Get engine
    automerging_engine = get_sentence_window_query_engine(
        index,
        similarity_top_k=similarity_top_k,
        rerank_model=rerank_model,
        rerank_top_n=rerank_top_n,
    )

    return automerging_engine


def basic_rag_setup(
    input_file: str,
    save_dir: str,
    llm_openai_model: SupportedOpenAIllms = "gpt-4o-mini",
    temperature: float = 0.1,
    embed_model: SupportedEmbedModels = "BAAI/bge-small-en-v1.5",
    similarity_top_k: int = 6,
    rerank_model: SupportedRerankModels = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    rerank_top_n: int = 2,
    **kwargs
):
    openai.api_key = get_openai_api_key()

    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

    save_dir = os.path.join(save_dir, "basic")
    if not os.path.exists(save_dir):
        document = load_data(input_file)
        index = VectorStoreIndex.from_documents([document])
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model)

    engine = index.as_query_engine(
        llm=OpenAI(model=llm_openai_model, temperature=temperature),
        similarity_top_k=similarity_top_k,
        node_postprocessors=[rerank],
    )
    return engine
