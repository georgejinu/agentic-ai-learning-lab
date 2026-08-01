"""Shared Gemini chat model factory (Vertex AI backend) for the learning lab.

All pattern demos in this repo call get_llm() instead of constructing their
own client, so the model/project/location live in one place.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and "
            "fill in your GCP project id (run `gcloud config get-value project` to find it)."
        )
    # gemini-2.5+ preview/GA models are only served on the global endpoint.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
