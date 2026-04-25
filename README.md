This repository is an example of an interactive chat application that uses Retrieval Augmented Generation to help solve the user's debugging issues

The code uses "FAISS" to vectorize inputs and find the proper context from the document based on the input question

Next, it generates the required outputs by feeding the proper context into the LLM (GPT-4o-mini)

To run the file, clone the repository, make sure you have a ".env" file within this directory and input OPENAI_API_KEY="YOUR KEY" in it

Next, run the command "python app.py" from the root directory and see the app on your browser by copy-pasting the proper flask app ip address 


