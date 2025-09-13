import json
from pathlib import Path

file_path = "../cryptol/examples/Karatsuba.cry"

# Path to your source code file
source_path = Path(file_path)

# Path to save JSON output
json_path = Path("data/source_code.jsonl")

# Read the source file as text
with open(source_path, "r", encoding="utf-8") as f:
    source_text = f.read()

# Wrap it in a dictionary so it's JSON-serializable
data = {"filename": str(source_path),"content": source_text}

# Write JSON to disk
with open(json_path, "w", encoding="utf-8") as f:
    #json.dumps(data, f, indent=2, ensure_ascii=False)
    f.write(json.dumps(data) + "\n")

print(f"Source code saved to {json_path}")
