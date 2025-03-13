import os
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler
import threading
import queue
llm = OllamaLLM(model="llama3.2", streaming=False, max_tokens=50, temperature=0.5)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

embeddings = OllamaEmbeddings(model="llama3.2")

vector_db = Chroma(embedding_function=embeddings, collection_name="nfsuchat", persist_directory="./chroma_db")

retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vector_db.as_retriever()
)

prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    AI: Provide a relevant answer without unnecessary details.
""")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

class TokenStreamHandler(BaseCallbackHandler):
    def __init__(self, stream_to_terminal=False):
        self.queue = queue.Queue()
    
    # def on_llm_new_token(self, token: str, **kwargs) -> None:
    #     self.queue.put(token)

    # def get_tokens(self):
    #     while True:
    #         token = self.queue.get()
    #         if token is None:
    #             break
    #         yield token


def chat_response(user_input, stream_to_terminal= False): 
    return chain.invoke({"question": user_input})["answer"]
    # handler = TokenStreamHandler(stream_to_terminal=stream_to_terminal)

    # thread = threading.Thread(target=chain, args=({"question": user_input}), kwargs={"callbacks":[handler]})
    # thread.start()

    # for token in handler.get_tokens():
    #     yield token
    
    # thread.join()
    # handler.queue.put(None)
    # method 3
    if len(user_input.split()) < 3:  # Short input (e.g., "Hello", "Hi")
        return llm.invoke(user_input)  # Skip context & history

    return chain.invoke({"question": user_input})["answer"]  