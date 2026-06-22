import re
import requests
import json


class LangGraphSupervisorEngine:
    def __init__(self):
        self.system_state_history = []

        # TIER 1: Edge Intent Dictionary (Runs in milliseconds)
        self.edge_intents = {
            r"\b(turn on|enable|start)\b.*\b(light|lights|ac|tv|appliance)\b": "DEVICE_ON",
            r"\b(turn off|disable|stop)\b.*\b(light|lights|ac|tv|appliance)\b": "DEVICE_OFF",
            r"\b(set|change)\b.*\b(temperature|temp)\b.*\b(\d+)\b": "SET_TEMP",
            r"\b(hi|hello|hey)\b": "GREETING"
        }

        # LOCAL MEMORY: Personalization Database (Fixes Fake #7)
        self.user_preferences = {
            "Garvit": "Prefers the AC set to 22 degrees and lights set to cool white.",
            "Alishri": "Prefers the AC set to 24 degrees and lights set to warm yellow.",
            "Unknown_Speaker": "Default home settings apply."
        }

    def _fast_intent_match(self, text):
        for pattern, intent in self.edge_intents.items():
            if re.search(pattern, text, re.IGNORECASE):
                return intent
        return None

    def _call_local_llm(self, prompt):
        """TIER 2: Makes a genuine HTTP request to a local Ollama instance (Fixes Fake #6)."""
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 20,  # Keep responses ultra-short to save xRT
                "temperature": 0.1
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"LLM Error: Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "[Offline Mode] Qwen 2.5 endpoint unreachable. Running edge heuristics only."

    def run_pipeline(self, speaker_id, transcript, context):
        cleaned_intent = transcript.strip().lower()
        response_state = {"is_approved": True, "matched_speaker": speaker_id}

        # 1. TIER-1 FAST ROUTING (Bypasses LLM for simple commands)
        matched_intent = self._fast_intent_match(cleaned_intent)

        if matched_intent:
            response_state[
                "final_execution_plan"] = f"TIER-1 NLU HIT -> Executing native edge command: [{matched_intent}] for {speaker_id}"
            self.system_state_history.append(response_state)
            return response_state

        # 2. TIER-2 HEAVY REASONING (The actual LLM call)
        # Inject personalization memory based on the identified speaker
        user_context = self.user_preferences.get(speaker_id, self.user_preferences["Unknown_Speaker"])

        llm_prompt = f"User '{speaker_id}' says: '{cleaned_intent}'. Profile: {user_context}. Provide a very brief, personalized smart-home response."

        llm_response = self._call_local_llm(llm_prompt)
        response_state["final_execution_plan"] = f"TIER-2 LLM RESPONSE -> {llm_response}"

        self.system_state_history.append(response_state)
        return response_state