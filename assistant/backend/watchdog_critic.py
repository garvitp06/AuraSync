import json
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM


class ValidationResult(BaseModel):
    is_safe: bool = Field(description="True if safe")
    reason: str = Field(description="Single sentence context reason")
    corrected_action: str = Field(description="Action token output")


class WatchdogCriticAgent:
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        print(f"[Watchdog Critic] Connecting to local SLM Engine ({model_name})...")
        self.llm = OllamaLLM(model=model_name, temperature=0.0)
        print("[Watchdog Critic] Compressed token safety agent live.")

    def validate_intent_stream(self, current_intent: str, speaker_id: str, system_context: dict) -> ValidationResult:
        """
        Hyper-fast token-compressed safety check mapping.
        """
        prior_speaker = system_context.get('last_speaker')
        prior_intent = system_context.get('last_intent')

        # Baseline chronological safety mapping
        if not prior_speaker or not prior_intent:
            return ValidationResult(
                is_safe=True,
                reason="Initial baseline command. System state clear.",
                corrected_action=current_intent
            )

        # Build a compressed token context window for the 1.5B model
        system_prompt = (
            "Task: Smart Home Safety Verification. Respond ONLY with valid JSON.\n"
            "Rules:\n"
            "- If prior intent was 'Turn off all appliances', block any different speaker trying to turn things on.\n"
            "- If no direct conflict exists, approve it.\n\n"
            f"Context:\n"
            f"- Prior Spk: {prior_speaker} | Prior Cmd: '{prior_intent}'\n"
            f"- Current Spk: {speaker_id} | Current Cmd: '{current_intent}'\n\n"
            "Output Format:\n"
            "{\n"
            '  "is_safe": true/false,\n'
            '  "reason": "Brief single-sentence explanation.",\n'
            '  "corrected_action": "Original command string or HOLDBACK"\n'
            "}"
        )

        try:
            response = self.llm.invoke(system_prompt)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            clean_json = response[json_start:json_end]
            data = json.loads(clean_json)

            return ValidationResult(
                is_safe=bool(data.get("is_safe", True)),
                reason=data.get("reason", "Approved"),
                corrected_action=data.get("corrected_action", current_intent)
            )
        except Exception as e:
            return ValidationResult(
                is_safe=True,
                reason="Pass through on internal parse error.",
                corrected_action=current_intent
            )