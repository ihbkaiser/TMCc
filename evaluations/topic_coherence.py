from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from itertools import combinations
from datasethandler.file_utils import split_text_word
import os
import tempfile

def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary,
                        topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return cv_per_topic, score


def TC_on_wikipedia(top_words: list[list[str]], cv_type: str = 'C_V', jar_dir = ".", wiki_dir = "."):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', dir='/tmp') as tmp_file:
        top_word_path = tmp_file.name
        for topic in top_words:
            tmp_file.write(" ".join(topic) + "\n")
    
    jar_dir = "/home/ducanh/Credit/TM-clusterrin/baseline/evaluations"
    wiki_dir = "/home/ducanh/Credit/TM-clusterrin"
    random_number = np.random.randint(100000)
    
    tmp_output_path = f"/tmp/tmp{random_number}.txt"
    cmd = (
        f"java -jar {os.path.join(jar_dir, 'pametto.jar')} "
        f"{os.path.join(wiki_dir, 'wikipedia_bd')} {cv_type} "
        f"{top_word_path} > {tmp_output_path}"
    )
    os.system(cmd)

    cv_score = []
    with open(tmp_output_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                score = float(parts[1])
                cv_score.append(score)
            except ValueError:
                print(f"[WARN] Skipped line (not a float): {line.strip()}")
                continue
    # os.remove(top_word_path)
    # os.remove(tmp_output_path)

    if len(cv_score) == 0:
        return [], 0.0
    
    return cv_score, sum(cv_score) / len(cv_score)