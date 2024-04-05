from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer("./e5-large-v2")
# query = "Represent this sentence for searching relevant passages: {lyrics}"

combined_string = """
The following is a summary of the lyrics:
{lyrics_summary}

It has the following keywords:
{lexical_keywords}

And the following emotional themes:
{sentiment_keywords}
"""

import json
from pydantic import BaseModel
from typing import List


class Lyrics(BaseModel):
    lyrics_summary: str
    lexical_keywords: List[str]
    sentiment_keywords: List[str]


def embed(doc: Lyrics):
    doc = combined_string.format(
        lyrics_summary=doc.lyrics_summary,
        lexical_keywords=", ".join(doc.lexical_keywords),
        sentiment_keywords=", ".join(doc.sentiment_keywords),
    )

    embeddings = model.encode(
        [doc],
        convert_to_numpy=True,
    )

    return embeddings


from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

api_keys = ["543c7086-c880-45de-8bce-6c9c906293bb"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embed/")  # , dependencies=[Depends(api_key_auth)])
async def embed_lyrics(doc: Lyrics):
    try:
        embedding = embed(doc)
        return json.dumps({"embedding": embedding.squeeze(0).tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
