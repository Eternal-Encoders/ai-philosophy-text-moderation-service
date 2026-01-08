from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="AI Philosophy Text Moderation Service",
    description="Filters non-philosophical or toxic input",
    version="1.0.0",
)

toxicity_classifier = pipeline(
    "text-classification",
    model="cointegrated/rubert-tiny-toxicity",
    tokenizer="cointegrated/rubert-tiny-toxicity",
)


topic_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

LABELS = ["philosophy", "not philosophy"]


class TextRequest(BaseModel):
    text: str


@app.post("/moderate")
def moderate_text(request: TextRequest):
    toxicity_result = toxicity_classifier(request.text, top_k=None)

    toxicity_dict = {item["label"]: item["score"] for item in toxicity_result}

    non_toxic = toxicity_dict.get("non-toxic", 0.0)
    dangerous = toxicity_dict.get("dangerous", 0.0)

    toxicity_score = 1 - non_toxic * (1 - dangerous)

    topic = topic_classifier(request.text, LABELS)
    philosophy_score = topic["scores"][0]

    if toxicity_score > 0.5:
        return {
            "allowed": False,
            "reason": "toxic",
            "toxicity_score": toxicity_score,
        }

    if philosophy_score < 0.5:
        return {
            "allowed": False,
            "reason": "off_topic",
            "topic_score": philosophy_score,
        }

    return {
        "allowed": True,
        "reason": "ok",
        "topic_score": philosophy_score,
        "toxicity_score": toxicity_score,
    }
