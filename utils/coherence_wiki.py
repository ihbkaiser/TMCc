import os
import numpy as np
import tempfile

def TC_on_wikipedia(top_words: list[list[str]], cv_type: str = 'C_V'):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', dir='/tmp') as tmp_file:
        top_word_path = tmp_file.name
        for topic in top_words:
            tmp_file.write(" ".join(topic) + "\n")
    
    jar_dir = "." 
    wiki_dir = "."  
    random_number = np.random.randint(100000)
    
    tmp_output_path = f"/tmp/tmp{random_number}.txt"
    cmd = (
        f"java -jar {os.path.join(jar_dir, 'palmetto.jar')} "
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



if __name__ == "__main__":
    test_topics = [
        ['word1', 'word2', 'word3'],
        ['word4', 'word5', 'word6'],
        ['word7', 'word8', 'word9'],
        ['word10', 'word11', 'word12'],
        ['word13', 'word14', 'word15'],
        ['word16', 'word17', 'word18'],
        ['word19', 'word20', 'word21'],
        ['word22', 'word23', 'word24'],
        ['word25', 'word26', 'word27'],
        ['word28', 'word29', 'word30']
    ]
    scores, avg = TC_on_wikipedia(test_topics, cv_type="C_V")
    print("\n=== FINAL RESULT ===")
    print("Scores:", scores)
    print("Average:", avg)
