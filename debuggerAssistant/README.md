This directory contains an example of an FastAPI interactive troubleshooting chat application using Langgraph:

## Features:
- RAG using stored previous conversation history
- Github MCP Server connection to refer to repositories during the troubleshooting session
- Qdrant vector database deployed on Azure Container Apps to vectorize inputs to increase RAG speed and accuracy

## Instructions:
- To run the file, clone the repository, make sure you have a ".env" file within this directory and input the following:

- OPENAI_API_KEY="YOUR KEY" 
- AZURE_STORAGE_CONNECTION_STRING="YOUR KEY"
- GITHUB_API_TOKEN = "YOUR TOKEN"

- Install required packages from requirements.txt using "pip install --no-cache-dir -r requirements.txt"

- Next, run the command "uvicorn main:app --host 0.0.0.0 --port 8000 --reload" from the root directory of the main.py file and see the app on your browser by copy-pasting the app ip address 


