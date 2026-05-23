import json

with open("data/processed/chunks_with_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

pso_chunks = [
    c for c in chunks
    if "PSO" in c.get("text", "") or "Particle Swarm" in c.get("text", "")
]

print(f"Chunks mentioning PSO: {len(pso_chunks)}")

for c in pso_chunks[:3]:
    print(f"\nCh{c.get('chapter_num')} | {c.get('section_heading')}")
    print(c['text'][:2000])