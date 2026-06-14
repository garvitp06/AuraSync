import time
import json
import random


def generate_mock_stream():
    """
    Simulates the continuous JSON output from the LRS3 Conformer + Voice ID Embedder.
    Alishri will use this to feed her LangGraph Supervisor.
    """
    mock_intents = [
        {"speaker_id": "Speaker_A", "transcript": "Turn on the living room TV", "confidence": 0.98},
        {"speaker_id": "Speaker_B", "transcript": "And make it louder", "confidence": 0.91},
        {"speaker_id": "Speaker_A", "transcript": "Wait, no, turn off the TV", "confidence": 0.88},  # Barge-in test
    ]

    print("--- AuraSync Edge Acoustic Pipeline Started ---")
    for intent in mock_intents:
        # Simulate processing delay on edge CPU (target xRT < 0.5)
        time.sleep(random.uniform(0.3, 0.6))

        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "speaker_id": intent["speaker_id"],
            "transcript": intent["transcript"],
            "confidence_score": intent["confidence"]
        }

        # Print JSON string directly to stdout for real-time stream pipe reading
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    generate_mock_stream()