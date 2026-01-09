from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_philosophy_text_allowed():
    """
    Проверяет, что корректный философский текст разрешается
    и проходит модерацию.
    """
    response = client.post(
        "/moderate",
        json={"text": "Что такое свобода воли в философии?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allowed"] is True
    assert "topic_score" in data
    assert "toxicity_score" in data


def test_toxic_text_blocked():
    """
    Проверяет, что токсичный текст блокируется.
    Например, оскорбление или угроза.
    """
    response = client.post(
        "/moderate",
        json={"text": "Ты тупой, идиот! Я тебя убью!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allowed"] is False
    assert data["reason"] == "toxic"
    assert data["toxicity_score"] > 0.5
