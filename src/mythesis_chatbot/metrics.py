# %%
from mythesis_chatbot import utils
import openai
from trulens.core import TruSession
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

data_dir = "../../data"

# %%
openai.api_key = utils.get_openai_api_key()

# %% Loading data

reader = SimpleDirectoryReader(input_dir=data_dir)
documents = reader.load_data()  # List of Document objects (one object per page)

# Merge into single document
document = Document(text="\n\n".join([doc.text for doc in documents]))


# %% Build sentence window index
index = utils.build_sentence_window_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    save_dir=os.path.join(data_dir, "sentence_index"),
)

# %%

sentence_window_engine = utils.get_sentence_window_query_engine(index)
