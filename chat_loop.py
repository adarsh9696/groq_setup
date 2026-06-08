import os
import time
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

file = open('C:/Users/adars/Documents/ai_module/sample_text.txt')
content = file.read()
file.close()

chunks = []
for c in content.split("\n\n"):
    if len(c.strip()) > 100:
        chunks.append(c.strip())

chunk_embeddings = model.encode(chunks)

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


def comparison(user_query) : 
    query_encoded = model.encode(user_query)
    scores = util.cos_sim(query_encoded, chunk_embeddings)

    top_results = torch.topk(scores[0], k=3)

    content = ''
    for idx in top_results.indices : 
        content = content+chunks[idx]+"\n\n---\n\n"
    return top_results,content
    """for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
    print(f"#{rank+1}: {sentences[idx]}")
    print(f"Score: {score:.4f}\n")  """

while prompt != "quit":

    prompt = input("Ask a question. To exit enter 'quit': ")

    if prompt == 'quit' : 
        break

    if not prompt.strip():
        continue

    retrieval_result,matches = comparison(prompt)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": filled_template+matches},
            *history[-6:],
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
    )

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    print("response is based on these top-3 retrieved sentences : ")
    for rank, (score, idx) in enumerate(zip(retrieval_result.values, retrieval_result.indices)):
        print(f"#{rank+1}: {chunks[idx]}")
        print(f"Score: {score:.4f}\n")

    print("Response : ")

    print(response.choices[0].message.content)
    time.sleep(2)