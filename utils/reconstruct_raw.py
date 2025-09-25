from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def reconstruct_raw(processed_train, processed_test, raw_full):
    # Fit TF-IDF on raw + processed
    vectorizer = TfidfVectorizer()
    tfidf_raw = vectorizer.fit_transform(raw_full)
    tfidf_train = vectorizer.transform(processed_train)
    tfidf_test = vectorizer.transform(processed_test)
    
    # Similarity: processed train vs raw full
    sim_train = cosine_similarity(tfidf_train, tfidf_raw)
    sim_test = cosine_similarity(tfidf_test, tfidf_raw)
    
    # Best matches
    train_indices = np.argmax(sim_train, axis=1)
    test_indices = np.argmax(sim_test, axis=1)
    
    raw_train = [raw_full[i] for i in train_indices]
    raw_test = [raw_full[i] for i in test_indices]
    
    return raw_train, raw_test, train_indices, test_indices

# # Example
# processed_train = ["cat dog", "fish bird"]
# processed_test = ["dog", "fish"]
# raw_full = ["this is cat and dog", "bird and fish are here", "dog runs fast", "fish swims"]

# raw_train, raw_test, train_idx, test_idx = reconstruct_raw(processed_train, processed_test, raw_full)

# print("Raw train:", raw_train)
# print("Raw test:", raw_test)
# print("Train indices:", train_idx)
# print("Test indices:", test_idx)