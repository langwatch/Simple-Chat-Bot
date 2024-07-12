from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import chainlit as cl

client = OpenAI()


import langwatch


@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(content=f"How can we help you??", disable_feedback=True)
    await msg.send()


@cl.on_message
async def main(message: cl.Message):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{message.content}"},
        ],
    )

    assistant_reply = response.choices[0].message.content

    await cl.Message(content=assistant_reply).send()
