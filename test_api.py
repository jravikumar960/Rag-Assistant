
import requests

API_URL = "http://127.0.0.1:8000"


def test_rag_api():
    # Test 1: Backend health
    health = requests.get(
        f"{API_URL}/health",
        timeout=10
    )
    assert health.status_code == 200
    print("PASS: Health endpoint")

    # Test 2: Normal question
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": "Explain Random Forest"},
        timeout=300
    )

    assert response.status_code == 200
    data = response.json()

    assert data["answer"].strip() != ""
    assert data["session_id"]
    print("PASS: Normal question")

    # Test 3: Source attribution
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

    first_source = data["sources"][0]
    assert first_source.get("filename")
    assert first_source.get("page") is not None
    print("PASS: Source attribution")

    # Test 4: Follow-up in the same session
    follow_up = requests.post(
        f"{API_URL}/ask",
        json={
            "question": "How does it reduce overfitting?",
            "session_id": data["session_id"]
        },
        timeout=300
    )

    assert follow_up.status_code == 200
    follow_data = follow_up.json()
    assert follow_data["answer"].strip() != ""
    assert follow_data["session_id"] == data["session_id"]
    print("PASS: Follow-up question")

    # Test 5: Follow-up without session context
    standalone = requests.post(
        f"{API_URL}/ask",
        json={"question": "How does it reduce overfitting?"},
        timeout=300
    )

    assert standalone.status_code == 200
    assert standalone.json()["answer"].strip() != ""
    print("PASS: Question without session context")


if __name__ == "__main__":
    test_rag_api()
    print("\nAll API checks passed!")
