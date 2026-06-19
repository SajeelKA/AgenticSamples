import os
import uuid
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    trim_messages,
)
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fixed: Import END to correctly terminate the graph state machine
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import traceable
import sqlite3
from datetime import datetime
import socket
import AzureModule

load_dotenv()
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
history = []
pending_answers = {}

profileChecker = AzureModule.AzureProfiles()
profileName = "guests"

llm = None
retriever = None
vectorstore = None
documents = None
mcp_tools = None
mcp_client = None
wf = None

            
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever, vectorstore, documents, mcp_tools, wf, mcp_client
    
    mcp_tools = await fetch_mcp_tools()   

    
    #url = "http://localhost:6333"
    url = "qdrant-container.gentlewater-60a1584b.eastus.azurecontainerapps.io:6333"
    
    client = QdrantClient(url=url)

    collections = [c.name for c in client.get_collections().collections]
    print('collections available: ', collections)

    # Bind the fetched MCP tools safely to the LLM instance
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(mcp_tools)

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    collection_name = "debugger"
    #profileName = socket.gethostname()
    
    # 1. Fetch initial profile logs once
    documents = profileRag(profileChecker, profileName)

    # 2. Vector DB Initialized safely without repeating chunks down the road
    vectorstore, retriever = initVectorDB(
        embeddings,
        url,
        collection_name,
        collections,
        documents
    )

    
    
    DB_PATH = "checkpoints_" + profileName + ".db"
    conn1 = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        wf = initGraph(saver, mcp_tools)
        yield
    conn1.close()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

initializationPrompt = SystemMessage(
    content=(
        "You are a senior software debugging assistant.\n\n"
        "Instructions:\n"
        "- Use previous conversation if relevant\n"
        "- Be precise and actionable\n"
        "- If debugging code, explain root cause + fix\n"
        "- Keep answers concise\n"
        "- You have automated GitHub tools available."
    )
)


        
def profileRag(profileChecker, profileName):
    d, _ = profileChecker.getUserId(profileName)
    fileList = profileChecker.files(d)
    logs = []
    for f in fileList:
        logs.append(
            Document(page_content=profileChecker.download(f).decode("utf-8"))
        )
    return logs

def initVectorDB(embeddings, url, collection_name, collections, docs):
    """Handles parsing and populating collection once at application startup."""
    if collection_name in collections:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )
    else:
        print('initiating new collection')
        # Move chunking and adding logic HERE so it runs exactly once on launch
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return vectorstore, retriever

async def fetch_mcp_tools():
    global mcp_client, mcp_tools

    """ mcp_client = MultiServerMCPClient({
        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={os.getenv('GITHUB_API_TOKEN')}",
                "ghcr.io/github/github-mcp-server"
            ],
            "transport": "stdio"
        }
    }) """
    
    mcp_client = MultiServerMCPClient({
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_API_TOKEN")
                },
                "transport": "stdio"
            }
        })
    
    mcp_tools = await mcp_client.get_tools()
    return mcp_tools

class ChatState(MessagesState):
    question: str
    context: list

def vectorSearch(state: ChatState):
    """Node strictly queries pre-existing database instead of rebuilding it."""
    docs = retriever.invoke(state["question"])
    relevant_context = "\n".join(d.page_content for d in docs)
    #print(docs)

    context_msg = SystemMessage(content=f"Relevant context logs:\n{relevant_context}")

    return {
        "context": [context_msg]
    }

@traceable(run_type="chain", name="Answer Question Node")
def answerQuestion(state: ChatState):
    full_messages = [initializationPrompt] + state.get("context", []) + state["messages"]

    messages = trim_messages(
        full_messages,
        strategy="last",
        max_tokens=2000,
        token_counter=llm,
    )

    response = llm.invoke(messages)
    return {"messages": [response]}

def initGraph(redis_saver, tools):
    g = StateGraph(ChatState)

    g.add_node("answerQuestion", answerQuestion)
    g.add_node("vectorSearch", vectorSearch)
    g.add_node("tools", ToolNode(tools))

    g.add_edge(START, "vectorSearch")   
    g.add_edge("vectorSearch", "answerQuestion")
    g.add_conditional_edges(
        "answerQuestion",
        tools_condition,  # Directs to "tools" if tool called, or END if finished
    )
    g.add_edge("tools", "answerQuestion")

    return g.compile(checkpointer=redis_saver)

@app.get("/")
async def home():
    return FileResponse("templates/index.html")

@app.get("/downloadChatHistory")
def downloadChatHistory():
    print('sending')
    
    return FileResponse(
        path=str(datetime.now())[:10] + '.txt',
        filename=str(datetime.now())[:10] + '.txt',
        media_type="application/octet-stream"
    )

    
@app.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form("default"),
    file: UploadFile | None = File(None),
):
    global documents

    file_content = ""
    if file:
        try:
            file_content = (await file.read()).decode("utf-8", errors="ignore")
        except Exception:
            file_content = "Could not read file."

    # If file was attached, append it directly to user context prompt
    user_content = question
    if file_content:
        user_content += f"\n\nAttached File Excerpt:\n{file_content[:1000]}"

    inputPrompt = HumanMessage(content=user_content)

    initial_input = {
        "messages": [inputPrompt],
        "question": question,
    }

    thread = {"configurable": {"thread_id": session_id}}
    answer = ""

    # Streaming the state machine execution loop 
    async for event in wf.astream(initial_input, thread, stream_mode="values"):
        answer = event["messages"][-1].content
    

    # Append tracking info
    history.append({"question": question, "answer": answer})
    q_id = str(uuid.uuid4())
    pending_answers[q_id] = {"question": question, "answer": answer}

    d, userIds = profileChecker.getUserId(profileName)
    profileChecker.addLogs(f"[Time:{datetime.now()}]\nQuestion: {question}\nAnswer: {answer}\n\n", f"""{d}/logs.txt""")
    #with open(f"{datetime.now():%Y-%m-%d}.txt", "a", encoding="utf-8") as f:
    #    f.write(f"Question: {question}\nAnswer: {answer}\n\n")

    return {"id": q_id, "answer": answer}
