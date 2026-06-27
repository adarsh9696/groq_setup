import os
import time
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
import json

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('sample_text.txt') as file: #C:/Users/adars/Documents/ai_module/sample_text.txt
    content = file.read()

chunks = []
for c in content.split("\n\n"):
    if len(c.strip()) > 100:
        chunks.append(c.strip())
updated_chunks = []
for idx in range(len(chunks)) :
    if idx > 0 :
        prefix = chunks[idx-1].split(". ")[-1] + " " 
    else : 
        prefix = ""
    if idx < len(chunks)-1 :
        suffix = " " + chunks[idx+1].split(". ")[0] 
    else : 
        suffix = ""
    updated_chunks.append(prefix + chunks[idx] + suffix)

chunk_embeddings = model.encode(updated_chunks)

prompt = 'start'

filled_template = """You are a document assistant. You have ONE job: answer questions using ONLY the document below. Moreover, you STRICTLY answer in a JSON format that you are provided with in this system prompt.

STRICT RULES :
- If the answer is in the document, answer it using only that information.
- If the answer is NOT in the document, respond with exactly: "I cannot answer this based on the provided document."
- Never use your training knowledge. Never answer from memory.
- No exceptions.
- Always provide a complete, detailed answer using all relevant information from the document.
- Do not give one-sentence answers when more relevant information is available.

STRICT FORMAT TO FOLLOW : 
{
    "answer": "the actual answer here",
    "confidence": "high/medium/low",
    "key_points": ["point 1", "point 2", "point 3"],
    "related_topics": ["topic 1", "topic 2"]
}


DOCUMENT:
"""

history = []

def comparison(user_query) : 
    query_encoded = model.encode(user_query)
    scores = util.cos_sim(query_encoded, chunk_embeddings)

    top_results = torch.topk(scores[0], k=3)
    count = 0
    for rank,value in enumerate(top_results.values) : 
        if value >= 0.3 : 
            count = rank +1
    
    if count == 0 : 
        return None,None
    
    top_results = torch.topk(scores[0], k=count)

    content = ''
    for idx in top_results.indices : 
        content = content+updated_chunks[idx]+"\n\n---\n\n"
    
    return top_results,content


while prompt != "quit":

    prompt = input("Ask a question. To exit enter 'quit': ")

    if prompt == 'quit' : 
        break

    if not prompt.strip():
        continue

    retrieval_result,matches = comparison(prompt)

    if matches is None : 
        print("I cannot find relevant information in the document.")
        continue

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
    test = response.choices[0].message.content
    test = test[test.index("{"):test.rindex("}")+1]
    parsed = json.loads(test)

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    """
    print("response is based on these top-3 retrieved sentences : ")
    for rank, (score, idx) in enumerate(zip(retrieval_result.values, retrieval_result.indices)):
        print(f"#{rank+1}: {updated_chunks[idx]}")
        print(f"Score: {score:.4f}\n")

    """

    print("Answer : \n"+ parsed["answer"]+"\n")
    print("Confidence : \n"+ parsed["confidence"]+"\n")

    print("Key points : \n")
    for keys in parsed["key_points"] : 
        print(" -"+keys)
    
    print("Related topics : \n")
    for topics in parsed["related_topics"] : 
        print(" -"+topics)
    time.sleep(2)