import pprint
import os
import google.generativeai as genai
import dotenv
import time


dotenv.load_dotenv()


class Gemini():
    def __init__(self, temperature=0):
        genai.configure(api_key=os.getenv("Google_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get_response(self, input):
        prompt = "\n\n".join([m["content"] for m in input])
        for i in range(10):
            try:
                response = self.model.generate_content(
                    prompt,
                    # generation_config=genai.types.GenerationConfig(
                    #     candidate_count=1,
                    #     max_output_tokens=2048,
                    #     temperature=0
                    # )
                )
                return response.text
            except Exception as e:
                print(f"Attempt {i+1} failed with error: {e}")
                time.sleep(2)
        
        return response.text[0]

