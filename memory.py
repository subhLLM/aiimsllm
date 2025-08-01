import threading

class ConversationMemory:
    def __init__(self, max_history_turns=10, summary_threshold=15):
        self.history = []
        self.contextual_entities = []
        self.current_topic = None
        self.last_entity_by_type = {}
        self.max_history_turns = max_history_turns
        self.summary_threshold = summary_threshold

    def add_turn(self, user_query, assistant_response, extracted_entities_map):
        turn_index = len(self.history)
        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "turn_index": turn_index,
            "entities": extracted_entities_map
        })

        if extracted_entities_map:
            for entity_type, entity_list in extracted_entities_map.items():
                for entity_value in entity_list:
                    if entity_value:
                        self.contextual_entities.append({
                            "value": entity_value,
                            "type": entity_type,
                            "turn_index": turn_index
                        })
                        self.last_entity_by_type[entity_type] = entity_value
            if self.contextual_entities:
                self.current_topic = self.contextual_entities[-1]

        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def get_last_entity_by_priority(self, type_priority=None):
        if not type_priority:
            type_priority = ["doctors", "departments", "rooms", "services", "buildings", "floors", "elevators", "opd", "ward", "office", "canteen"]

        if isinstance(type_priority, str):
            type_priority = [type_priority]

        for entity_type in type_priority:
            for turn in reversed(self.history):
                entities = turn.get("entities", {})
                if entities and entity_type in entities and entities[entity_type]:
                    return entities[entity_type][-1]
        return None

    def get_contextual_history_text(self, num_turns=5):
        history_text = ""
        recent_turns = self.history[-num_turns:]

        for turn in recent_turns:
            user_msg = turn.get("user", "[no user input]")
            assistant_msg = turn.get("assistant", "[no assistant response]")
            turn_index = turn.get("turn_index", -1)
            history_text += f"User (Turn {turn_index}): {user_msg}\nAssistant (Turn {turn_index}): {assistant_msg}\n"

        return history_text.strip()

    def get_relevant_entities_from_recent_turns(self, turns_to_check=3):
        relevant = []
        if not self.history or not self.contextual_entities:
            return []
        
        current_turn_index = self.history[-1]["turn_index"]
        
        for entity_info in reversed(self.contextual_entities):
            if (current_turn_index - entity_info["turn_index"]) < turns_to_check:
                if not any(r['value'] == entity_info['value'] and r['type'] == entity_info['type'] for r in relevant):
                    relevant.append(entity_info)
            else:
                break
        return list(reversed(relevant))

class InMemoryUserMemoryStore:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def get(self, user_id):
        with self.lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = ConversationMemory()
            return self.sessions[user_id]

    def save(self, user_id, memory: ConversationMemory):
        with self.lock:
            self.sessions[user_id] = memory

    def clear(self, user_id):
        with self.lock:
            self.sessions.pop(user_id, None)

    def all_user_ids(self):
        return list(self.sessions.keys())