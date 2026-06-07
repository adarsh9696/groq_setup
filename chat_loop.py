import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

personas = [
    {"name": "Professor", "prompt": "You are a professor. Explain concepts formally and in depth."},
    {"name": "Kids teacher", "prompt": "You are explaining to a 10 year old. Use simple words and a fun analogy."},
    {"name": "JSON only", "prompt": "Respond in JSON only. Fields: summary, example, difficulty_level."},
]

prompt = input("Enter a topic: ")

for persona in personas:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": persona["prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
    )

    print(f"\n===== {persona['name']} =====")
    print(response.choices[0].message.content)