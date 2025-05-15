from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, load_tools, Tool, AgentType
from dotenv import load_dotenv
import os

load_dotenv()


with open("prompts/commonsense_agent_prompt.txt", "r", encoding="utf-8") as f:
    commonsense_agent_prompt_template = f.read()

def commonsense_agent(task, history=None):

    prompt = commonsense_agent_prompt_template % (task, history if history else "None")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("API_KEY")
    )

    # # Initialize the agent with tools
    # agent = initialize_agent(
    #     llm = llm,
    #     # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=False
    # )

    # result = agent.run(prompt)

    result = llm.invoke(prompt).content.strip()

    # Extract final answer  

    return result.strip()
