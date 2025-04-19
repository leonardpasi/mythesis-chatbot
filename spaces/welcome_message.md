## Welcome!

*⚠️ This is a private demo using my own OpenAI API key. Please use responsibly. ⚠️*

This chatbot uses retrieval augmented generation (RAG) to answer questions about topics
that are discussed in my master thesis. My master thesis can be found on the [GitHub repo](https://github.com/leonardpasi/mythesis-chatbot) of the project.

Here you get to choose between three RAG techniques:
- **classic retrieval** (which includes a reranker model, so it's actually not the simplest RAG imaginable)
- **sentence window retrieval**
- **auto-merging retrieval**

Feel free to experiment with different modes! Note that a little extra delay is to be expected when switching to another mode.
Also, note that all your queries (as well as system responses, and evaluation of these responses) are automatically logged on a remote PostgreSQL database for continuous monitoring of the deployed systems.

Each of these systems has been optimized for performance by doing a grid search on the
relevant parameters. Performance is quantified with five metrics:
- **context relevance**: is the retrieved context relevant to the query?
- **groundedness**: is the response supported by the context?
- **answer relevance**: is the response relevant to the query?
- **cost**
- **latency**
