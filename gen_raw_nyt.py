from openai import OpenAI
from dotenv import load_dotenv
import os
from time import sleep


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

processed_train_path = "/home/ducanh/Credit/TM-clusterrin/tm_datasets/NYT/train_texts.txt"
processed_test_path = "/home/ducanh/Credit/TM-clusterrin/tm_datasets/NYT/test_texts.txt"

output_path = "/home/ducanh/Credit/TM-clusterrin/tm_datasets/NYT/raw_texts.txt"

with open(processed_train_path, "r", encoding="utf-8") as f:
    processed_train = f.readlines()

# Read all lines
with open(processed_test_path, "r", encoding="utf-8") as f:
    processed_test = f.readlines()

nyt_text = processed_train + processed_test
# formatted_text = "\n".join(nyt_text[:10])

SYSTEM_PROMPT = """
    TASK
    Rewrite the given token sequence into a fluent English sentence or phrase.

    RULES
    1. Keep all tokens exactly as written and in the same order.
    2. Do not reorder, delete, or alter tokens.
    3. Add only minimal helper words or punctuation if needed.
    4. Output only the final sentence in a single line (no quotes, no explanations, no extra line breaks).

    STYLE
    - Neutral tone, present tense.
    - Capitalize normally, but preserve the tokens' original casing.
    """

# TEMPLATE = f"""
#     TASK
#     Given a short token sequence, rewrite it as a natural, grammatical English sentence or phrase.

#     HARD RULES
#     1) Keep ALL given tokens exactly as written and in the SAME ORDER.
#     2) Do NOT reorder, delete, or change the given tokens (no stemming/inflection changes).
#     3) Output ONLY the final sentence (no quotes, no explanations, no escape letters such as \n).

#     STYLE
#     - Default to neutral tone and present tense.
#     - Capitalize normally; preserve the tokens' original casing inside the sentence.
#     """

with open(output_path, "w", encoding="utf-8") as out_f:
    for i in range(len(nyt_text)):
    # for i in range(3):
        formatted_text = nyt_text[i]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"INPUT:\n{formatted_text}"}
            ],
            temperature=0.4
        )

        output_text = response.choices[0].message.content.strip()

        # Write each result on its own line
        out_f.write(output_text + "\n")

        print(f"Processed line {i + 1}: {output_text}")
        # sleep(5)

print(f"✅ All results saved to {output_path}")