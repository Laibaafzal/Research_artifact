import json

with open("relation_output.json", "r") as f:
    re_data = json.load(f)

with open("subtitle_ner_output.json", "r") as f:
    ner_data = json.load(f)

id2entities = {
    item["id"]: {ent["word"]: ent["label"] for ent in item.get("entities", [])}
    for item in ner_data
}

relation_type_constraints = {
    "AFFECTS": [("ACT", "PER"), ("EMO", "PER"), ("ACT", "LOC")],
    "ASSOCIATED_WITH": [("PER", "PER"), ("PER", "ORG"), ("ORG", "LOC")],
    "BELIEVES_IN": [("PER", "SPIRIT")],
    "FEELS": [("PER", "EMO")],
    "GUIDES": [("PER", "PER")],
    "HAPPENED_IN": [("ACT", "LOC"), ("DATE", "LOC"), ("TIME", "LOC")],
    "HAPPENED_ON": [("ACT", "DATE"), ("ACT", "TIME")],
    "HAS_ROLE": [("PER", "ROLE")],
    "HOSTS": [("ORG", "PER"), ("LOC", "PER")],
    "LOCATED_IN": [("PER", "LOC"), ("ORG", "LOC"), ("LOC", "LOC")],
    "PERFORMS": [("PER", "ACT")],
    "SPEAKS_WITH": [("PER", "PER")],
    "TRAVELS_TO": [("PER", "LOC")],
}

def is_valid_relation(item):
    ent_map = id2entities.get(item["id"], {})
    head_type = ent_map.get(item["head"])
    tail_type = ent_map.get(item["tail"])
    if not head_type or not tail_type:
        return False
    valid_types = relation_type_constraints.get(item["relation"])
    if not valid_types:
        return False
    return (head_type, tail_type) in valid_types

filtered_relations = [item for item in re_data if is_valid_relation(item)]

with open("relation_output_filtered.json", "w") as f:
    json.dump(filtered_relations, f, indent=2)

print(f"✅ Done! {len(filtered_relations)} out of {len(re_data)} relations kept.")
print("📄 Saved to: relation_output_filtered.json")
