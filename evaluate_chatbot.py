import json
import csv
from datetime import datetime
from tqdm import tqdm
from main22spacy import chat  # ensure chat() is importable
from sklearn.metrics import classification_report

# Load test dataset
with open("hospital_intents.jsonl", "r", encoding="utf-8") as f:
    test_cases = [json.loads(line.strip()) for line in f]

results = []
intent_preds = []
intent_trues = []

exact_entity_matches = 0
partial_entity_matches = 0
response_quality_pass = 0

for test in tqdm(test_cases):
    query = test["input"]
    expected_intent = test.get("expected_intent")
    expected_entities = test.get("expected_entities", {})
    expected_keywords = test.get("expected_response_contains", [])
    user_id = "test_eval_user"

    try:
        response = chat(query, user_id)
    except Exception as e:
        print(f"Error calling chat() for query: {query}")
        print(e)
        continue

    debug = response.get("debug_info", {})
    predicted_intent = debug.get("detected_task_type", "")
    extracted_entities = debug.get("extracted_entities", {})
    final_answer = response.get("answer", "")
    processing_time = debug.get("processing_time_seconds", 0)

    # Intent Accuracy
    intent_preds.append(predicted_intent)
    intent_trues.append(expected_intent)
    intent_correct = (predicted_intent == expected_intent)

    # Entity Matching (basic scoring)
    entity_exact = True
    entity_partial = False
    for key, expected_vals in expected_entities.items():
        predicted_vals = extracted_entities.get(key, [])
        if isinstance(predicted_vals, str):
            predicted_vals = [predicted_vals]
        if set(predicted_vals) != set(expected_vals):
            entity_exact = False
            if any(val in predicted_vals for val in expected_vals):
                entity_partial = True

    if entity_exact:
        exact_entity_matches += 1
    elif entity_partial:
        partial_entity_matches += 1

    # Response Quality Check
    answer_lower = final_answer.lower()
    response_quality_passed = all(kw.lower() in answer_lower for kw in expected_keywords)
    if response_quality_passed:
        response_quality_pass += 1

    results.append({
        "query": query,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "intent_correct": intent_correct,
        "expected_entities": expected_entities,
        "extracted_entities": extracted_entities,
        "entity_exact_match": entity_exact,
        "entity_partial_match": entity_partial,
        "expected_keywords": expected_keywords,
        "answer": final_answer,
        "response_quality_passed": response_quality_passed,
        "response_time_sec": round(processing_time, 2)
    })

# === Summary Metrics ===
print("\n=== Evaluation Summary ===")
print(classification_report(intent_trues, intent_preds, zero_division=0))

print(f"Entity Exact Matches: {exact_entity_matches}/{len(test_cases)}")
print(f"Entity Partial Matches: {partial_entity_matches}/{len(test_cases)}")
print(f"Response Quality Passes: {response_quality_pass}/{len(test_cases)}")

# === Save to CSV ===
timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"eval_report_{timestamp}.csv"

with open(csv_filename, "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Evaluation complete. CSV saved to: {csv_filename}")
