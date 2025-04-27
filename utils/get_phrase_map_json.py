import json

# 1. Load your word-level vocabulary: word → word_id
word2id = {}
with open('datasets/20NG/vocab.txt', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        w = line.strip()
        if w:
            word2id[w] = idx

# 2. Load your phrase vocabulary: phrase → phrase_id
phrase2id = {}
with open('datasets/20NG/vocab_phrase.txt', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        p = line.strip().lower()
        if p:
            phrase2id[p] = idx

# 3. Build the mapping: phrase_id → [word_id1, word_id2]
phrase_map = {}
for phrase, pid in phrase2id.items():
    words = phrase.split()   # e.g. ["new","york"]
    # look up each word in word2id
    ids = [word2id[w] for w in words if w in word2id]
    phrase_map[str(pid)] = ids

# 4. Save to JSON
with open('datasets/20NG/phrase_map.json', 'w', encoding='utf-8') as f:
    json.dump(phrase_map, f, indent=2)

print(f"Saved phrase_map.json with {len(phrase_map)} entries.")
