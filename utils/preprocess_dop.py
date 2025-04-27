# preprocess document of phrases
#!/usr/bin/env python3
"""
Script: extract_phrases.py

Extracts only the defined phrases from each document line.
Usage:
    python extract_phrases.py \
        --docs /path/to/train_texts.txt \
        --vocab /path/to/vocab_phrase.txt \
        --output /path/to/output_phrases.txt
"""
import re
import argparse

def load_phrases(vocab_file):
    """
    Load phrases from vocab_file, normalize to lowercase,
    and sort by descending word length to prioritize longer phrases.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        phrases = [line.strip().lower() for line in f if line.strip()]
    # Sort by number of words (descending) to match longest phrases first
    phrases.sort(key=lambda p: len(p.split()), reverse=True)
    return phrases


def extract_phrases(text, phrases):
    """
    Given a document text and list of phrases,
    return all phrases found in the text.
    Uses word-boundary regex to avoid partial matches.
    """
    text_lower = text.lower()
    found = []
    for phrase in phrases:
        # Build regex pattern to match whole phrase
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text_lower):
            found.append(phrase)
            # Remove matched phrase to prevent overlapping matches
            text_lower = re.sub(pattern, ' ', text_lower)
    return found


def process_files(docs_file, vocab_file, output_file):
    phrases = load_phrases(vocab_file)
    with open(docs_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            doc = line.strip()
            if not doc:
                fout.write('\n')
                continue
            extracted = extract_phrases(doc, phrases)
            # Join extracted phrases with space or any delimiter you prefer
            fout.write(' '.join(extracted) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract defined phrases from each document line."
    )
    parser.add_argument('--docs', default="datasets/20NG/train_texts.txt", help="Path to documents file")
    parser.add_argument('--vocab', default="datasets/20NG/vocab_phrase.txt", help="Path to vocab phrases file")
    parser.add_argument('--output', default="datasets/20NG/train_dop.txt", help="Path to output file")
    args = parser.parse_args()

    process_files(args.docs, args.vocab, args.output)
