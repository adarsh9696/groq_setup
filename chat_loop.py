import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()   # reads .env and injects into os.environ

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt = "continue"

history = []

while prompt != "quit" :
    prompt = input("enter prompt. to stop type 'quit' : ")

    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": f"{prompt}"},*history
    ],
    temperature=1,
    max_tokens=512,
    )   

    history.append({"role": "user", "content": f"{prompt}"})
    history.append({"role": "assistant", "content": f"{response.choices[0].message.content}"})

    print(response.choices[0].message.content)
