from serpapi import GoogleSearch
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv
from langchain.agents import initialize_agent, load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load the search agent prompt template
with open("prompts/search_agent_prompt.txt", "r", encoding="utf-8") as f:
    search_agent_prompt_template = f.read()

def search(task, history=None):
    # Format prompt with task and history
    prompt = search_agent_prompt_template % (task, history if history else "None")

    # Initialize LLM and tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("API_KEY")
    )
    tools = load_tools(["serpapi"], serpapi_api_key=os.getenv("SERPAPI_KEY"))

    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    result = agent.run(prompt)

    # Extract final answer
    if isinstance(result, str):
        for line in result.splitlines():
            if line.strip().startswith("Final Answer:"):
                return line.split("Final Answer:")[-1].strip()
        return result.strip()

    return result