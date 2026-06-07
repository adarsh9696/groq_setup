import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

file = open('C:/Users/adars/Documents/ai_module/sample_text.txt')
content = file.read()
file.close()

content = content.strip()

prompt = 'start'

filled_template = """You are a document assistant. You have ONE job: answer questions using ONLY the document below.

STRICT RULES:
- If the answer is in the document, answer it using only that information.
- If the answer is NOT in the document, respond with exactly: "I cannot answer this based on the provided document."
- Never use your training knowledge. Never answer from memory.
- No exceptions.

DOCUMENT:
"""

history = []

while prompt != "quit":

    prompt = input("Ask a question. To exit enter 'quit': ")

    if prompt == 'quit' : 
        break

    if not prompt.strip():
        continue

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": filled_template+content},
            *history[-6:],
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
    )

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    print(response.choices[0].message.content)
    time.sleep(2)