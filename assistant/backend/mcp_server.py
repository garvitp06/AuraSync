import json


class MockMCPServer:
    """
    Simulates a Local Model Context Protocol (MCP) server controlling smart home appliances
    with zero cloud latency.
    """

    def __init__(self):
        # Map tool names to their execution functions
        self.tool_registry = {
            "set_temperature": self._set_temperature,
            "toggle_appliance": self._toggle_appliance,
            "lock_door": self._lock_door
        }

    def execute_tool(self, payload_json: str) -> dict:
        """Parses the JSON payload from the Execution Agent and triggers the local device."""
        try:
            payload = json.loads(payload_json)
            tool_name = payload.get("tool")
            kwargs = payload.get("parameters", {})

            if tool_name not in self.tool_registry:
                return {"status": "error", "message": f"Tool '{tool_name}' not recognized by MCP Server."}

            # Execute the routed function
            return self.tool_registry[tool_name](**kwargs)

        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON payload structure."}

    # --- Simulated Hardware Interfaces ---

    def _set_temperature(self, device_id: str, temperature: int) -> dict:
        print(f"🌡️ [HARDWARE EXECUTION] Setting {device_id} to {temperature}°C")
        return {"status": "success", "device": device_id, "state": temperature}

    def _toggle_appliance(self, device_id: str, state: str) -> dict:
        print(f"🔌 [HARDWARE EXECUTION] Toggling {device_id} to {state.upper()}")
        return {"status": "success", "device": device_id, "state": state}

    def _lock_door(self, device_id: str, is_locked: bool) -> dict:
        action = "LOCKED" if is_locked else "UNLOCKED"
        print(f"🚪 [HARDWARE EXECUTION] {device_id} is now {action}")
        return {"status": "success", "device": device_id, "state": action}


if __name__ == "__main__":
    server = MockMCPServer()
    print("✅ Local MCP Server Initialized.")

    # Simulate an incoming payload from the LangGraph Execution Agent
    sample_payload = json.dumps({
        "tool": "set_temperature",
        "parameters": {
            "device_id": "living_room_ac",
            "temperature": 22
        }
    })

    print("\nSimulating Execution Payload:")
    result = server.execute_tool(sample_payload)
    print("Server Response:", result)