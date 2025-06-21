"""
Acknowledgement:
    Based on the code from
    https://github.com/dominiksinsaarland/evaluating-topic-model-output
"""

import os
import json
import numpy as np
import random
import time
from gemini_model import Gemini 

llm_client = Gemini(temperature=0.0)

def get_system_prompt(dataset_name: str) -> str:
    """
    Return json system prompt for LLM evaluation of topic model output.
      {
        "rating": 1|2|3,
        "explanation": "Let’s think step by step: …"
      }
    """
    base = (
        "You are a helpful assistant evaluating the top words of a topic model output for a given topic. "
        "Please output **only** a JSON object with two keys:\n"
        '  - "rating": an integer 1, 2, or 3\n'
        '  - "explanation": a reasoning string that **starts** with "Let’s think step by step: "\n'
        "The JSON must be the only content in your response.\n"
    )
    if dataset_name == "20NG":
        return base + (
            "The topic modeling is based on the 20 Newsgroups dataset, a collection of approximately "
            "20,000 newsgroup posts across 20 forums."
        )
    elif dataset_name == "BBC_new":
        return base + (
            "The topic modeling is based on the BBC News dataset (2,225 articles in business, entertainment, politics, sport, tech)."
        )
    elif dataset_name == "NYT":
        return base + (
            "The topic modeling is based on the New York Times Annotated Corpus (articles from 1987–2007)."
        )
    elif dataset_name.startswith("WOS"):
        return base + (
            "The topic modeling is based on the Web of Science dataset (scholarly records across disciplines)."
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def llm_rating(tm_dataset_root_dir: str,  # path to tm_dataset
               beta_dir: str, # path to beta.npy
               dataset_name: str,   #dataset_name e.g., "20NG", "BBC_new", "NYT", "WOS"
               seed = 42,
               num_words: int = 10,
               sample_k: int = 10,
               sleep_sec: float = 0.1):
    random.seed(seed)


    vocab_path = os.path.join(tm_dataset_root_dir, "vocab.txt")
    with open(vocab_path, "r", encoding="utf-8") as vf:
        vocab_list = [w.strip() for w in vf if w.strip()]
    vocab = {i: w for i, w in enumerate(vocab_list)}

    beta_path = os.path.join(beta_dir, "beta.npy")
    beta = np.load(beta_path)  
    num_topics = beta.shape[0]

    top_words = []
    for row in beta:
        top_idxs = row.argsort()[::-1][:num_words]
        top_words.append([vocab[idx] for idx in top_idxs])

    system_prompt = get_system_prompt(dataset_name)


    results = []
    sampled_ids = random.sample(range(num_topics), k=sample_k)
    for tid in sampled_ids:
        words = top_words[tid].copy()
        random.shuffle(words)
        user_prompt = ", ".join(words)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
        text = llm_client.get_response(messages)
        text = text.strip()

        try:
            payload = json.loads(text)
            rating = int(payload.get("rating", None))
            assert rating in [1, 2, 3], f"Invalid rating: {rating}, expected 1, 2, or 3"
            explanation = payload.get("explanation", "")
        except Exception:
            rating = None
            explanation = text

        results.append({
            "path": beta_dir,
            "dataset_name": dataset_name,
            "topic_id": tid,
            "user_prompt": user_prompt,
            "rating": rating,
            "explanation": explanation
        })
        time.sleep(sleep_sec)

    return results