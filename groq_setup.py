

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()   # reads .env and injects into os.environ

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Explain what you are in 3 sentences."}
    ],
    temperature=1,
    max_tokens=512,
)

print(response.choices[0].message.content)