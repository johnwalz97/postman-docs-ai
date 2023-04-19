import os
from typing import List

import dotenv
import openai
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

dotenv.load_dotenv()

app = FastAPI()
# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

qdrant_endpoint = os.getenv("QDRANT_ENDPOINT")
qdrant_client = QdrantClient(host="localhost", port=6333)

tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


class SearchRequest(BaseModel):
    query: str


@app.post("/search", response_model=str)
async def search(request: SearchRequest):
    query_embedding = get_embedding(request.query.strip())
    context_matches = search_qdrant(query_embedding)
    context = build_context_string(context_matches)

    return get_openai_completion(request.query.strip(), context)


def get_embedding(text):
    # openai recommends relacing newlines with spaces
    response = openai.Embedding.create(
        input=text.replace("\n", " "),
        model="text-embedding-ada-002",
    )
    return response["data"][0]["embedding"]


def search_qdrant(query_embedding):
    response = qdrant_client.search(
        collection_name="doc-embeddings",
        query_vector=query_embedding,
        limit=10,
    )
    # filter out results with a score less than 0.78 (arbitrary)
    return [match.payload for match in response if match.score > 0.78]


def build_context_string(context_matches):
    total_tokens = 0
    context = ""

    for match in context_matches:
        context_str = match["content"].strip() + "\n---\n"

        total_tokens += len(tiktoken_encoding.encode(context_str))
        if total_tokens > 1500:
            break

        context += context_str

    return context


def get_openai_completion(query, context) -> str:
    messages = [
        {
            "role": "system",
            "content": """You are a very enthusiastic Postman AI who loves to help people! Given the following information from the Postman documentation, answer the user's question using only that information, outputted in markdown format.
If you are unsure and the answer is not explicitly written in the documentation, say "Sorry, I don't know how to help with that."
Always include related code snippets if available.""",
        },
        {
            "role": "user",
            "content": f"Here is the documentation for Postman:\n{context}",
        },
        {
            "role": "user",
            "content": """Answer my next question using only the above documentation.
You must also follow the below rules when answering:
-Do not make up answers that are not provided in the documentation.
-If you are unsure and the answer is not explicitly written in the documentation context, say "Sorry, I don't know how to help with that."
-Prefer splitting your response into multiple paragraphs.
-Output as pretty markdown with code snippets if available.""",
        },
        {
            "role": "user",
            "content": f"Here is my question:\n{query}",
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
