import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from dexscreener import DexscreenerClient
import os

# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

system_prompt_template = """You are an AI agent for general web3 questions. You cannot respond to requests outside of web3.
if asked anything about daos.fun use these data:
```
What is OnlyFi?
OnlyFi is the #1 decentralized exchange (DEX) built natively on the OnlyLayer Blockchain, a Layer 2 blockchain. It combines AI-driven tools with advanced Layer 2 technology to offer users faster transactions, lower fees, and smarter trading solutions.

What problem is OnlyFi trying to solve?
OnlyFi addresses key challenges in decentralized finance (DeFi), including high transaction fees, slow transaction speeds, and the lack of intelligent trading tools for users. By leveraging Layer 2 scalability and AI, OnlyFi creates a seamless, efficient, and cost-effective trading experience.

How does OnlyFi work?
OnlyFi operates on the OnlyLayer Blockchain, a high-performance Layer 2 platform. It enables users to trade tokens, provide liquidity, and earn rewards, all while benefiting from low transaction fees (0.15%) and lightning-fast speeds. Its AI tools help optimize trades and provide actionable insights for better decision-making.

How is AI implemented in OnlyFi?
AI is integrated into OnlyFi to enhance the trading experience by offering features such as predictive market analysis, smart order routing, and risk management tools. These AI-driven insights empower users to make smarter trading decisions with minimal effort.

Does OnlyFi support multichain?
Yes, OnlyFi is designed to support multichain functionality, enabling seamless integration with other blockchain ecosystems and providing users access to a broader range of tokens and liquidity pools.

What is the transaction fee on OnlyFi?
OnlyFi offers one of the lowest transaction fees in the industry, charging just 0.15% per transaction, making it an affordable option for traders and liquidity providers.

Is OnlyFi safe to use?
Yes, OnlyFi prioritizes security with robust smart contract audits, decentralized architecture, and advanced security protocols. Users can trade and provide liquidity with confidence, knowing their assets are protected.
```
"""
chat_prompts = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_prompt_template)
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="{input}", input_variables=["input"])
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate(chat_prompts)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# tools
api_wrapper = DuckDuckGoSearchAPIWrapper()
search_desc = "search tool based on duckduckgo, useful to check for questions you can't answer, but not token details. input should be a search query."
search = DuckDuckGoSearchResults(api_wrapper=api_wrapper, source="news", description=search_desc)

# dexscreener tool
def dex_tool_func(token_name: str):
    client = DexscreenerClient()
    search_results = client.search_pairs(token_name)

    return search_results

dex_tool = Tool(name="dexscreener tool",
                func=dex_tool_func,
                description="Tool to use when users ask question about any token, input should be function name")

tools = [search, dex_tool]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    st.title("OnlyFi AI")
    # chat interface for consistent queries
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def prepare_chat_history(messages):
        chat_history = []
        for message, kind in messages:
            if kind == "ai":
                message = AIMessage(message)
            elif kind == "user":
                message = HumanMessage(message)
            chat_history.append(message)
        return chat_history

    # Display for all the messages
    for message, kind in st.session_state.messages:
        with st.chat_message(kind):
            st.markdown(message)

    prompt = st.chat_input("Ask your questions ...")

    if prompt:
        # Handling prompts and rendering to the chat interface
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            [prompt, "user"]
        )

        with st.spinner("Generating response"):
            try: 
                chat_history = prepare_chat_history(st.session_state.messages)
                output = agent_executor.invoke(
                    {"input": prompt, "chat_history": chat_history}
                )['output']
                
            
                st.chat_message("ai").markdown(output)
                st.session_state.messages.append([output, "ai"])
            except Exception as e:
                error_message = "I am sorry, I am currently unable to help with that. Can you rephrase?"
                st.chat_message("ai").write(error_message)
                st.session_state.messages.append([error_message, "ai"])
                print(e)