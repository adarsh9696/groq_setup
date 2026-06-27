from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from dotenv import load_dotenv


class Question(BaseModel):
    question: str




@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    memory = ConversationBufferWindowMemory(
        k=3,  # only keep last 3 exchanges
        memory_key="chat_history",
        return_messages=True
    )

    loader = TextLoader("sample_text.txt") #C:/Users/adars/Documents/ai_module/sample_text.txt
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists ("FAISS_index") : 
        vectorstore = FAISS.load_local("faiss_index",embeddings, allow_dangerous_deserialization=True)
    else : 
        vectorstore = FAISS.from_documents(chunks,embeddings)
        vectorstore.save_local("faiss_index")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    custom_prompt = PromptTemplate(
        template="""You are a document assistant. Answer ONLY from the context below.
    If the answer is not in the context, say "I cannot answer this based on the provided document."

    Context: {context}
    Chat History: {chat_history}
    Question: {question}

    Answer:""",
        input_variables=["context", "chat_history", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    app.state.chain = qa_chain
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
def read_item(query: Question, request: Request) :
    response = request.app.state.chain.invoke({"question" :query.question})
    return {"answer": response["answer"]}