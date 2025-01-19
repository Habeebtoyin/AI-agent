from tkinter import E
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
For DAO Token holders
How It Works
1. Fundraise
Creators have 1 week to fundraise the desired SOL amount. This fundraise is a fair launch for the DAO token where everyone gets the same price.
Creators choose who to invite to the Party Round. You might get an invite link on their Telegram, X, or other communities.
2. Trading (Fundraise successful)
Once fundraise is over, creators take charge of 90 percent of SOL to invest on their favorite Solana protocols, and the token goes public on an AMM with 10% of fundraised SOL. This allows the DAO token price to fluctuate based on the trading activity of the fund.
Once launched, you can sell your DAO tokens anytime, because the pool liquidity is locked on DAOS.FUN pool.
3. Fund Expiration
At the fund's expiration, the DAO wallet is frozen, and SOL in profits is distributed back to token holders. You can burn your DAO tokens to redeem the DAO’s underlying assets, or simply sell it on the curve anytime.
Frequently Asked Questions
1. Why are DAO tokens mintable?
The DAO token mint authority is set to the voting module (coming Q4 2024-Q1 2025). Once the voting module is released, DAO token holders can can vote to mint more tokens to fundraise, to pay for DEX, or permanently burn the mint authority.
2. What is an investment DAO
A creator-funded smart wallet with special rules that invests on behalf of DAO token holders.
3. What are verified creators?
Creators that we extensively verify will have a blue checkmark next to them. Any creator without a checkmark you will have to trust at your own risk. DYOR.
4. Can I sell the DAO token at anytime?
Yes, as long as the market cap of the DAO token exceeds the original fundraise amount.
5. What happens if creator does not meet fundraising goal within week? 
You can redeem your SOL back if fundraising fails.
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
    st.title("Oculus AI")
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