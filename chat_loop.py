import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

role = input("enter role for the template : ")

audience = input("enter audience for the template : ")

template = """You are a {role} explaining things to a {audience}.

Here are some examples of how you should respond:

User: What is gravity?
Assistant: Gravity is a fundamental physical interaction that causes all entities with mass or energy to attract one another.  It is the force responsible for keeping planets in orbit, holding atmospheres in place, and giving weight to physical objects on planetary surfaces.

User: What is photosynthesis?
Assistant: Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert light energy from the sun into chemical energy stored in sugars.  This process primarily occurs in the chloroplasts of plant cells, where the pigment chlorophyll captures sunlight to drive the reaction.

Always follow this exact style and length in your responses."""

filled_template = template.format(role=role, audience=audience)

prompt = 'start'

while prompt != "quit":

    prompt = input("Enter a topic. To exit enter 'quit': ")

    if prompt == 'quit' : 
        break

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": filled_template},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
    )

    
    print(response.choices[0].message.content)