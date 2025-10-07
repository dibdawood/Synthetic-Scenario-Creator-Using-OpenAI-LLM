import os
from apikey import apikey

import streamlit as st 
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

st.title('Synthetic Scenario Creator - Job Role')
prompt = st.text_input('Input Box - Enter a Job Role (Role 1):') 

# PROMPT template
colleague_prompt_template = PromptTemplate(
    input_variables = ['role'],
    template = 'Mention one job role who works the most with {role} in an organization. Answer in one or two words max.'
)

scenario_prompt_template = PromptTemplate(
    input_variables = ['role', 'colleague', 'wiki_research_role_self', 'wiki_research_role_colleague'],
    template = 'Create a possible scenario between two roles. Role_1: {role}, Role_2: {colleague}, Role_Description_1: {wiki_research_role_self}, Role_Description_2: {wiki_research_role_colleague}.'
)


# Memory
colleague_memory = ConversationBufferMemory(input='role', memory_key='chat_history')
scenario_memory = ConversationBufferMemory(input_key='role', memory_key='chat_history')

#llms
llm = OpenAI(temperature=0.9)
colleague_chain = LLMChain(llm=llm, prompt=colleague_prompt_template, verbose=True, output_key='colleague', memory=colleague_memory)
scenario_chain = LLMChain(llm=llm, prompt=scenario_prompt_template, verbose=True, output_key='scenario', memory=scenario_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to screen if there's a prompt
if (prompt):
    colleague = colleague_chain.run(prompt)
    wiki_research_role_self = wiki.run(prompt)
    wiki_research_role_colleague = wiki.run(colleague)
    scenario = scenario_chain.run(role=prompt,colleague=colleague, wiki_research_role_self=wiki_research_role_self, wiki_research_role_colleague=wiki_research_role_colleague)

    st.write('Colleague (Role-2): ', colleague)
    st.write('Scenario: ', scenario)

    with st.expander('Colleague History'):
        st.info(colleague_memory.buffer)

    with st.expander('Scenario History'):
        st.info(scenario_memory.buffer)
    
    with st.expander('Wiki Research on Role 1'):
        st.info(wiki_research_role_self)

    with st.expander('Wiki Research on Role 2'):
        st.info(wiki_research_role_colleague)