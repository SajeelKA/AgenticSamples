
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
import os
import uuid

from flask import render_template
import dotenv

chat_histories = {}

dotenv.load_dotenv()

embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

app = Flask(__name__)

def get_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]
    
# Root route
@app.route("/")
def home():
    return render_template("index.html")

# In-memory storage (persist later if needed)
documents = []
vectorstore = None

# Temporary store for unanswered Qs
pending_answers = {}

# --- Helper: rebuild vectorstore ---
def rebuild_vectorstore():
    global vectorstore
    if documents:
        vectorstore = FAISS.from_documents(documents, embedding_model)

# --- Prompt template ---
prompt = ChatPromptTemplate.from_template("""
You are a senior software debugging assistant.

Conversation so far:
{history}

Relevant past fixes:
{context}

Current question:
{question}

Instructions:
- Use previous conversation if relevant
- Be precise and actionable
- If debugging code, explain root cause + fix
- Keep answers concise

Answer:
""")

# --- ASK endpoint ---
@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore

    data = request.json
    question = data.get("question")
    session_id = data.get("session_id", "default")
    history = get_history(session_id)
    
    context = ""

    # Retrieve similar past solutions if available
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

    # Generate response
    chain = prompt | llm
    
    history_text = "\n".join([
        f"User: {h['question']}\nAssistant: {h['answer']}"
        for h in history[-5:]   # last 5 turns only
    ])
    
    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    answer = response.content
    history_text = "\n".join([
        f"User: {h['question']}\nAssistant: {h['answer']}"
        for h in history[-5:]   # last 5 turns only
    ])
    
    # --- Save to history ---
    history.append({
        "question": question,
        "answer": answer
    })
    # Store temporarily (only persist if user confirms)
    q_id = str(uuid.uuid4())
    pending_answers[q_id] = {
        "question": question,
        "answer": answer
    }

    return jsonify({
        "id": q_id,
        "answer": answer
    })

# --- FEEDBACK endpoint ---
@app.route("/feedback", methods=["POST"])
def feedback():
    global documents

    data = request.json
    q_id = data.get("id")
    success = data.get("success")

    if q_id not in pending_answers:
        return jsonify({"error": "Invalid ID"}), 400

    if success:
        qa = pending_answers[q_id]

        # Store as RAG knowledge
        doc = Document(
            page_content=f"Problem: {qa['question']}\nSolution: {qa['answer']}",
            metadata={"source": "user_confirmed"}
        )

        documents.append(doc)
        rebuild_vectorstore()

    # Remove from pending
    del pending_answers[q_id]

    return jsonify({"status": "recorded"})

if __name__ == "__main__":
    app.run(debug=True)

# when send button is pressed, "ask" function is called and question is passed
# vector_store like FAISS or weaviate are used to retrieve vectors from past documents (will be used as context)
# prompt_from_template object is piped to the llm to create the "chain"
# next the question and context will be passed to chain by using "invoke"
#answer is put into JSON object and returned



"""

<!DOCTYPE html>
<html>
<head>
    <title> Investigate Issue </title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }
        .chat-container {
            width: 600px;
            margin: 40px auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .messages {
            height: 400px;
            overflow-y: auto;
            border-bottom: 1px solid #ddd;
            margin-bottom: 10px;
        }
        .message {
            margin: 10px 0;
        }
        .user {
            text-align: right;
            color: blue;
        }
        .bot {
            text-align: left;
            color: green;
        }
        .input-box {
            display: flex;
        }
        input {
            flex: 1;
            padding: 10px;
        }
        button {
            padding: 10px;
        }
        .feedback {
            margin-top: 5px;
        }
    </style>
</head>
<body>

<div class="chat-container">
    <h2>Investigate Issue</h2>
    <div class="messages" id="messages"></div>

    <div class="input-box">
        <input id="question" placeholder="Ask a debugging question..." />
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
async function sendMessage() {
    const input = document.getElementById("question");
    const text = input.value;
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: text})
    });

    const data = await res.json();

    addBotMessage(data.answer, data.id);
}

function addMessage(text, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message " + sender;
    msgDiv.innerText = text;
    document.getElementById("messages").appendChild(msgDiv);
}

function addBotMessage(text, id) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot";

    const content = document.createElement("div");
    content.innerText = text;

    const feedback = document.createElement("div");
    feedback.className = "Helpful?";

    const up = document.createElement("button");
    up.innerText = "Solved?";
    up.onclick = () => sendFeedback(id, true);

    const down = document.createElement("button");
    down.innerText = "Still going";
    down.onclick = () => sendFeedback(id, false);

    feedback.appendChild(up);
    feedback.appendChild(down);

    msgDiv.appendChild(content);
    msgDiv.appendChild(feedback);

    document.getElementById("messages").appendChild(msgDiv);
}

async function sendFeedback(id, success) {
    await fetch("/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: id, success: success})
    });

    alert("Thanks for the feedback! Please enter next question if required");
}
</script>

</body>
</html>


"""


