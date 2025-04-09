import os

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


def build_automerging_index(
    documents,
    llm,
    embed_model,
    save_dir,
    chunk_sizes=[2048, 512, 128],
):

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    Settings.llm = llm
    Settings.node_parser = node_parser
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
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


def rag_setup(data_dir: str):

    openai.api_key = get_openai_api_key()

    # 1. Load data
    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()  # List of Document objects (one object per page)
    # Merge into single document
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # 2. Build index
    index = build_sentence_window_index(
        [document],
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        save_dir=os.path.join(data_dir, "sentence_index"),
    )

    # 3. Get engine
    sentence_window_engine = get_sentence_window_query_engine(index)

    return sentence_window_engine
