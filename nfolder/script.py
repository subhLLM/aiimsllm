import json

# Load the original JSON data
with open('sample.json', 'r') as f:
    data = json.load(f)

# Process each room/feature
for feature in data:
    if "lifts" in feature and feature["lifts"]:
        # Keep only the lift with the minimum distance
        min_lift = min(feature["lifts"], key=lambda x: x.get("distance", float('inf')))
        feature["lifts"] = [min_lift]

    if "stairs" in feature and feature["stairs"]:
        # Keep only the stair with the minimum distance
        min_stair = min(feature["stairs"], key=lambda x: x.get("distance", float('inf')))
        feature["stairs"] = [min_stair]

# Save the modified data into a new JSON file
with open('filtered_sample.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Filtered JSON saved as 'filtered_sample.json'")
