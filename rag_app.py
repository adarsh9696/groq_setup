import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

st.title("My RAG built with Langchain")
st.write("This app uses FAISS search and ChromaDB vectorstore")

if "chain" not in st.session_state : 
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    )

    memory = ConversationBufferWindowMemory(
    k=3,  # only keep last 3 exchanges
    memory_key="chat_history",
    return_messages=True
    )

    loader = TextLoader("C:/Users/adars/Documents/ai_module/sample_text.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists ("faiss_index") : 
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
    
    st.session_state["chain"] = qa_chain
    st.session_state["messages"] = []

for message in st.session_state.messages : 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# BLOCK 3 — runs only when user submits input
if prompt := st.chat_input("Ask a question about the document"):

    # display user bubble immediately
    with st.chat_message("user"):
        st.write(prompt)

    # call your chain
    response = st.session_state.chain.invoke({"question": prompt})
    answer = response["answer"]

    # display assistant bubble
    with st.chat_message("assistant"):
        st.write(answer)

    # save both to session_state so they persist across reruns
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})