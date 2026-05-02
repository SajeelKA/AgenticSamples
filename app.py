
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
import os
import uuid
from datetime import datetime
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

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

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

Uploaded file content:
{file_content}

Instructions:
- Use previous conversation if relevant
- Be precise and actionable
- If debugging code, explain root cause + fix
- Keep answers concise

Answer:
""")

# invoke llm
@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore

    question = request.form.get("question")
    session_id = request.form.get("session_id", "default")
    file = request.files.get("file")
    history = get_history(session_id)
    history_length = 5
    
    file_content = ""

    if file:
        try:
            file_content = file.read().decode("utf-8", errors="ignore")
        except Exception:
            file_content = "Could not read file."
    max_file_len = 1000
    file_content = file_content[:max_file_len]
    
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
        for h in history[-history_length:]   # last 5 turns only
    ])
    
    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question,
        "file_content": file_content
    })

    answer = response.content
    history_text = "\n".join([
        f"User: {h['question']}\nAssistant: {h['answer']}"
        for h in history[-history_length:]   # last 5 turns only
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
    
    with open(str(datetime.now())[:10] + '.txt', 'a') as f:
        f.write('Question: {0}\nAnswer: {1}\n'.format(question, answer) )
    
    return jsonify({
        "id": q_id,
        "answer": answer
    })

# register feedback
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
    app.run(debug=True)#host="0.0.0.0", port=80
#    app.run(host="0.0.0.0", port=80)
    

