import json
from pathlib import Path

source = Path(r"TokenEconomyDataset_V1.json")
target = Path(r"hf_prompts.jsonl")

def load_json_allow_comments(p: Path):
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))

data = load_json_allow_comments(source)

with target.open("w", encoding="utf-8") as out:
    for item in data["prompts"]:
        out_obj = {
            "prompt_id": item.get("prompt_id"),
            "category": item.get("category"),
            "type": item.get("type"),
            "prompt": item.get("prompt"),
            "criteria": (item.get("criteria") or [""])[0]  # first element
        }
        out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

print(f"Wrote {target}")