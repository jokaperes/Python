chave_openai = "insira sua chave da API da open ai aqui"

#pip install langchain

#pip install openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    HumanMessage,
    SystemMessage
)

llm = ChatOpenAI(openai_api_key=chave_openai, temperature=0.9, model_name="gpt-3.5-turbo")

response = llm([HumanMessage(content="Qual a capital do Brasil?")])

print(response)

#pip install transformers
#pip install textract
#pip install faiss-cpu

import os
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

import textract
doc = textract.process("content/Bitcoin.pdf")

with open('content/Bitcoin.txt', 'w', encoding="utf-8") as f:
    f.write(doc.decode('utf-8'))

with open('content/Bitcoin.txt', 'r', encoding="utf-8") as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings(openai_api_key=chave_openai, model="text-embedding-ada-002")

db = FAISS.from_documents(chunks, embeddings)

query = "O que é Blockchain?"
docs = db.similarity_search(query)
docs[0]

chain = load_qa_chain(OpenAI(openai_api_key=chave_openai, temperature=0), chain_type="stuff")

chain.run(input_documents=docs, question=query)