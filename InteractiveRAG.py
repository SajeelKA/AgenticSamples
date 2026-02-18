import os
import requests
from typing import List, TypedDict

import dotenv


from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END

import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_community.vectorstores import Weaviate
import warnings
import logging

from typing import TypedDict, List

class RAGGraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str    

def getAnswer(st, llm, retriever):
    documents = retriever.invoke(st['question'])
    
    template = """You are an assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, say that you don't know.
                    Use three sentences maximum and keep the answer concise. 
                    Don't mention anything as a nursery rhyme.
                    
                    Question: {question}
                    Context: {context}
                    Answer:
                    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({
        "question": st['question'],
        "context": context
    })

    return generation
    
#make sure to put OPENAI_API_KEY=<KEY> in your .env file in this directory
dotenv.load_dotenv()

url = 'https://raw.githubusercontent.com/SajeelKA/AgenticSamples/refs/heads/main/resource.txt'
res = requests.get(url)

with open("resource1.txt", "w") as f:
    f.write(res.text)


loader = TextLoader("./resource1.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    client=client,
    by_text=False,
)

retriever = vectorstore.as_retriever()

st = RAGGraphState()

llm = ChatOpenAI(
    model="gpt-4o-mini",  # modern replacement for gpt-3.5-turbo
    temperature=0
)

while 1:
    st['question'] = input("\n== what's your question about Jack and Jill? ==\n")
    print('> ' , getAnswer(st, llm, retriever))
    again = input("\nDo you want to ask another question? (Enter 'y' to confirm)\n ")
    if again != 'y':
        print("\nbye\n")
        break
        
