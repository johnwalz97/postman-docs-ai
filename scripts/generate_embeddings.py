import json
import os
import re
import sys
from glob import glob

import dotenv
import openai

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_doc(markdown_content):
    _, metadata, content = re.split(r"---\s*\r?\n", markdown_content, maxsplit=2)

    title = metadata.splitlines()[0].split("title: ")[1].strip("\"")
    page_id = metadata.splitlines()[2].split("page_id: ")[1].strip("\"")

    return title, page_id, [
        ("## " if i else "") + s for i, s in enumerate(content.split("## "))
        if not s.startswith("Contents")
    ]


def get_embedding(text):
    # response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    # return response["data"][0]["embedding"]
    return [0, 1, 2, 3, 4]


def process_directory(root_dir):
    pages = []
    embeddings = []

    for file_path in glob(f"{root_dir}/**/*.md", recursive=True):
        print(f"Processing {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title, page_id, chunks = parse_doc(content)

        pages.append({
            "page_id": page_id,
            "title": title,
            "file_path": file_path,
            "content": content,
        })

        for chunk in chunks:
            embeddings.append({
                "page_id": page_id,
                "content": chunk,
                "embedding": get_embedding(chunk),
            })

    return pages, embeddings


def main(root_dir: str):
    pages, embeddings = process_directory(root_dir)
    print(f"Generated {len(embeddings)} embeddings from {len(pages)} pages")

    with open("embeddings.json", "w") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embeddings.py <root_dir>")
        sys.exit(1)

    main(sys.argv[1])
