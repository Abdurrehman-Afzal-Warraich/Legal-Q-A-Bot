import chromadb

def create_collection(name, persist_path):
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(name=name)
    return collection

def query_collection(collection, query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results
