#Importing the Deps 
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey

#Prompt Templates
title_template = PromptTemplate(
    input_variables = ["topic"],
    template = "Write me a title for an article about {topic} "
)

script_template = PromptTemplate(
    input_variables = ["title", "wikipedia_research"],
    template = "Write me an article based on this title TOPIC: {title} while leveraging this wikipedia reserch:{wikipedia_research}"
)
#App framework
st.title("First Langchain Service")
prompt = st.text_input("Enter your prompt here")

#Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")


#llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm =llm, prompt= title_template, verbose = True,
                    output_key="title", memory=title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose=True, 
                    output_key = "script", memory=script_memory)



wiki = WikipediaAPIWrapper()

#Show response to the screen if there's a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research = wiki_research)

    st.write(title)
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Script History"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research History"):
        st.info(wiki_research)