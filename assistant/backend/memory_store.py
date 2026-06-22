import chromadb
import json

# Initialize a persistent client saving data locally
CHROMA_DATA_PATH = "./chroma_data"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Get or create the collection for resident context
collection = client.get_or_create_collection(name="resident_profiles")


def initialize_profiles():
    """Seeds the local ChromaDB with baseline user preferences."""
    profiles = [
        {
            "speaker_id": "Speaker_A",
            "preferences": {"living_room_temp": 22, "music_genre": "indie pop", "lights": "warm_white"}
        },
        {
            "speaker_id": "Speaker_B",
            "preferences": {"living_room_temp": 18, "music_genre": "classical", "lights": "cool_white"}
        }
    ]

    for profile in profiles:
        collection.upsert(
            documents=[json.dumps(profile["preferences"])],
            metadatas=[{"speaker_id": profile["speaker_id"]}],
            ids=[profile["speaker_id"]]
        )
    print("✅ Local ChromaDB resident profiles initialized.")


def get_resident_context(speaker_id: str) -> dict:
    """Retrieves the context and preferences for a specific speaker."""
    results = collection.get(ids=[speaker_id])

    if results and results['documents']:
        return json.loads(results['documents'][0])
    return {"error": "Profile not found."}


if __name__ == "__main__":
    initialize_profiles()
    # Test the retrieval
    print("Testing Retrieval for Speaker_A:", get_resident_context("Speaker_A"))