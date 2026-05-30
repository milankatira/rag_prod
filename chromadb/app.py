import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("test_collection")

documents = [
    {"id": "1", "text": "Hello, world!"},
    {"id": "2", "text": "This is a test document."},
    {"id": "3", "text": "This is another test document."},
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

results = collection.query(
    query_texts=["Hello, world!"],
    n_results=2,
)

print(results)
