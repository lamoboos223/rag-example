from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # Use HuggingFace embeddings instead of Bedrock
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'},  # Use GPU if available
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings