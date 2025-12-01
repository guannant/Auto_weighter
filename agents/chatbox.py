import openai
from openai import APIStatusError, InternalServerError
import time

MODEL = "argo:gpt-5-mini"

client = openai.OpenAI(
    api_key="whatever+random",
    base_url="http://0.0.0.0:60963/v1",
)
def openai_chat_completion(prompt, model=MODEL, temperature=0.3, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
            )   
            return resp.choices[0].message.content.strip()
        except (APIStatusError, InternalServerError) as e:
            # Retry on 5xx
            if getattr(e, "status_code", 500) >= 500:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except Exception as e:
            # Retry if server returned HTML instead of JSON
            if "unexpected mimetype" in str(e).lower():
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("ChatCompletion failed after retries")