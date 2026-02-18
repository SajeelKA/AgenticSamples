This repository is an example of Retrieval Augmented Generation which uses a file with the "Jack and Jill" nursery rhyme as the external data resource

The code uses "Weavile" to vectorize inputs and find the proper context from the document based on the input question

Next, it generates the required outputs by feeding the proper context into the LLM (GPT-4o-mini)

To run the file, clone the repository, make sure you have a ".env" file within this directory and input OPENAI_API_KEY=<KEY> in it

Next, run the command "python InteractiveRAG.py" from the root directory

