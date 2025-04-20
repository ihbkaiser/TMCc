# phrase_utils.py — Utilities to extract and save phrase collocations
"""
Usage:
  python phrase_utils.py --data_path datasets/20NG --out_dir datasets/20NG
"""
import os
import re
import string
import argparse
import numpy as np
import scipy.sparse
from gensim.models.phrases import Phrases
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from file_utils import read_text
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_punct = ''.join(sorted(set(string.punctuation) - {"'"}))
_replace = re.compile(f'[{re.escape(_punct)}]')

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r"\S+@\S+", ' ', text)
    text = text.replace('_', ' ')
    text = _replace.sub(' ', text)
    text = text.replace("'", '')
    return re.sub(r'\s+', ' ', text).strip()


def mine_phrases(
    train_txt: str,
    test_txt: str,
    vocab_txt: str,
    min_count: int = 5,
    threshold: float = 10.0,
    delimiter: str = ' ',
    out_dir: str = '.',
):
    os.makedirs(out_dir, exist_ok=True)

    train_docs = [clean_text(line) for line in read_text(train_txt)]
    test_docs  = [clean_text(line) for line in read_text(test_txt)]
    all_docs = train_docs + test_docs


    tokenized = [simple_preprocess(strip_tags(doc), deacc=True) for doc in all_docs]

    phrases_model = Phrases(sentences=tokenized, min_count=min_count, threshold=threshold, delimiter=delimiter)
    phrases_dict = phrases_model.find_phrases(tokenized)
    all_phrases = list(phrases_dict.keys())

    vocab = set(read_text(vocab_txt))
    filtered = [p for p in all_phrases if all(w in vocab for w in p.split())]
    filtered.sort()
    Vp = len(filtered)


    with open(os.path.join(out_dir, 'vocab_phrase.txt'), 'w', encoding='utf-8') as f:
        for phrase in filtered:
            f.write(phrase + '\n')

    word2idx = {w:i for i,w in enumerate(read_text(vocab_txt))}
    pairs = [[word2idx[w1], word2idx[w2]] for w1,w2 in (p.split(delimiter) for p in filtered)]
    np.save(os.path.join(out_dir, 'phrase_pairs.npy'), np.array(pairs, dtype=np.int64))

    def build_phrase_bow(docs):
        B = len(docs)
        bow = np.zeros((B, Vp), dtype=np.float32)
        for i, doc in enumerate(docs):
            text = doc  
            for j, phrase in enumerate(filtered):
                bow[i,j] = text.count(phrase.replace(delimiter, ' '))
        return bow

    train_bow = build_phrase_bow(train_docs)
    test_bow  = build_phrase_bow(test_docs)
    scipy.sparse.save_npz(os.path.join(out_dir, 'train_phrase.npz'), scipy.sparse.csr_matrix(train_bow))
    scipy.sparse.save_npz(os.path.join(out_dir, 'test_phrase.npz'),  scipy.sparse.csr_matrix(test_bow))

    print(f"Mined {Vp} phrases; outputs saved under {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Mine and save phrase collocations")
    parser.add_argument('--data_path', type=str, default="datasets/20NG",
                        help='Root folder containing train_texts.txt, test_texts.txt, vocab.txt')
    parser.add_argument('--out_dir',    type=str, default='datasets/20NG',
                        help='Directory to save phrase files')
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--delimiter', type=str, default=' ')
    args = parser.parse_args()

    train_txt = os.path.join(args.data_path, 'train_texts.txt')
    test_txt  = os.path.join(args.data_path, 'test_texts.txt')
    vocab_txt = os.path.join(args.data_path, 'vocab.txt')
    mine_phrases(
        train_txt,
        test_txt,
        vocab_txt,
        min_count=args.min_count,
        threshold=args.threshold,
        delimiter=args.delimiter,
        out_dir=args.out_dir,
    )
