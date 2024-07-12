from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import chainlit as cl

client = OpenAI()

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader

import langwatch


pdf_path = "./docs/support.pdf"
loader = PyPDFLoader(file_path=pdf_path)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text = text_splitter.split_documents(documents=doc)

embeddings = OpenAIEmbeddings()

vectorStore = FAISS.from_documents(text, embeddings)
vectorStore.save_local("vectors")
loaded_vectors = FAISS.load_local("vectors", embeddings)

chat_history = ChatMessageHistory()


@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(content=f"How can we help you??", disable_feedback=True)
    await msg.send()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=chat_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=loaded_vectors.as_retriever(),
        memory=memory,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
@langwatch.trace()
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(
        message.content,
        callbacks=[cb, langwatch.get_current_trace().get_langchain_callback()],
    )

    answer = res["answer"]

    await cl.Message(content=answer).send()
