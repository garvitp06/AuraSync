import sqlite3
import os

DB_PATH = 'safety_store.db'


def init_db():
    """Initializes the SQLite database and pre-populates device safety rules."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create the safety rules table
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS device_rules
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       device_name
                       TEXT
                       NOT
                       NULL,
                       restricted_intent
                       TEXT
                       NOT
                       NULL,
                       conflict_condition
                       TEXT
                       NOT
                       NULL,
                       resolution_action
                       TEXT
                       NOT
                       NULL
                   )
                   ''')

    # Clear existing rules to avoid duplicates on re-runs
    cursor.execute('DELETE FROM device_rules')

    # Pre-populate with Phase 1 safety policies
    rules = [
        ('living_room_tv', 'volume_up', 'concurrent_volume_down', 'BLOCK_AND_NOTIFY'),
        ('smart_oven', 'turn_on', 'occupancy_0', 'BLOCK_SAFETY_RISK'),
        ('front_door', 'unlock', 'night_mode_active', 'REQUIRE_PIN_VERIFICATION')
    ]

    cursor.executemany('''
                       INSERT INTO device_rules (device_name, restricted_intent, conflict_condition, resolution_action)
                       VALUES (?, ?, ?, ?)
                       ''', rules)

    conn.commit()
    conn.close()
    print("✅ Safety Policy Store initialized successfully.")


def check_hardware_conflict(current_intent: str, device_name: str, active_states: list) -> dict:
    """
    Queries the database to verify if an intent violates constraints.
    active_states represents current environment variables (e.g., ['occupancy_0', 'concurrent_volume_down'])
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
                   SELECT conflict_condition, resolution_action
                   FROM device_rules
                   WHERE device_name = ?
                     AND restricted_intent = ?
                   ''', (device_name, current_intent))

    rules = cursor.fetchall()
    conn.close()

    for conflict_condition, resolution_action in rules:
        if conflict_condition in active_states:
            return {
                "status": "BLOCKED",
                "reason": f"Violates safety policy: {conflict_condition}",
                "action": resolution_action
            }

    return {"status": "APPROVED", "reason": "No hardware conflicts detected."}


# Run initialization if executed directly
if __name__ == "__main__":
    init_db()