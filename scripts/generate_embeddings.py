import json
import os
import re
import sys
from glob import glob
from uuid import uuid4

import dotenv
import openai
from qdrant_client import QdrantClient, models
from tqdm import tqdm

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_doc(markdown_content):
    _, metadata, content = re.split(r"---\s*\r?\n", markdown_content, maxsplit=2)

    title = None
    page_id = None

    for line in metadata.splitlines():
        if line.startswith("title: "):
            title = line.split("title: ")[1].strip('"')
        elif line.startswith("page_id: "):
            page_id = line.split("page_id: ")[1].strip('"')

    if not title:
        raise ValueError("Could not parse title from metadata")

    return (
        title,
        page_id,
        [
            ("## " if i else "") + s
            for i, s in enumerate(content.split("## "))
            if not s.startswith("Contents")
        ],
    )


def get_embedding(text):
    # openai recommends relacing newlines with spaces
    response = openai.Embedding.create(
        input=text.replace("\n", " "),
        model="text-embedding-ada-002",
    )
    return response["data"][0]["embedding"]


def process_directory(root_dir):
    pages = []
    embeddings = []

    print(f"Processing {root_dir}...")

    for file_path in tqdm(glob(f"{root_dir}/**/*.md", recursive=True)):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title, page_id, chunks = parse_doc(content)

        pages.append(
            {
                "page_id": page_id,
                "title": title,
                "file_path": file_path,
                "content": content,
            }
        )

        for chunk in chunks:
            embeddings.append(
                {
                    "page_id": page_id,
                    "file_path": file_path,
                    "embedding_id": str(uuid4()),
                    "content": chunk,
                    "embedding": get_embedding(chunk),
                }
            )

    return pages, embeddings


def load_embeddings():
    with open("data/embeddings.json", "r") as f:
        embeddings = json.load(f)

    embedding_len = len(embeddings[0]["embedding"])

    client = QdrantClient(host="localhost", port=6333)
    client.recreate_collection(
        collection_name="doc-embeddings",
        vectors_config=models.VectorParams(
            size=embedding_len,
            distance=models.Distance.DOT,
        ),
    )

    print("Loading embeddings into Qdrant...")

    for embedding in tqdm(embeddings):
        client.upsert(
            collection_name="doc-embeddings",
            points=[
                models.PointStruct(
                    id=embedding["embedding_id"],
                    vector=embedding["embedding"],
                    payload={
                        "page_id": embedding["page_id"],
                        "content": embedding["content"],
                    },
                )
            ],
        )

    print(f"Loaded {len(embeddings)} embeddings into Qdrant.")


def main(root_dir: str):
    pages, embeddings = process_directory(root_dir)
    print(f"Generated {len(embeddings)} embeddings from {len(pages)} pages")

    with open("data/pages.json", "w") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    with open("data/embeddings.json", "w") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)


def test_search():
    print("Testing search...")

    client = QdrantClient(host="localhost", port=6333)

    query = "how do i reset my password"
    query_embedding = get_embedding(query)

    response = client.search(
        collection_name="doc-embeddings",
        query_vector=query_embedding,
        limit=1,
    )

    print(response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embeddings.py <root_dir>")
        sys.exit(1)

    main(sys.argv[1])

    load_embeddings()

    test_search()
