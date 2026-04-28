import json
import time
import os
from openai import OpenAI

# Define the system prompt
SYS_PROMPT = """You are given a very general task. Now, you should pretend to be a user and give answer to a related query. You should provide a response to the query to show your preferences or requirements.

Please directly provide a concise and coherent response."""

# Load the API key
if os.path.exists("./secret.json"):
    client = OpenAI(
        api_key=json.load(open("./secret.json"))["api_key"],
        base_url=json.load(open("./secret.json"))["base_url"]
    )
else:
    client = OpenAI(api_key="sk-...")


def gpt_chatcompletion(messages, model="gpt-4o"):
    """
    Perform GPT chat completion with retries.
    """
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                n=1,
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")

if __name__ == "__main__":
    print(gpt_chatcompletion([{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": "### Task\n\n### Query\nWhat is your favorite color?\n\n### Response\n"}]))

