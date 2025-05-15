from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, load_tools, Tool, AgentType
from dotenv import load_dotenv
import os

load_dotenv()

# Load the math agent prompt template
with open("prompts/code_agent_prompt.txt", "r", encoding="utf-8") as f:
    code_agent_prompt_template = f.read()

def code_agent(task, history=None):
    # Format the prompt with task and history
    prompt = code_agent_prompt_template % (task, history if history else "None")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("API_KEY")
    )


    # Initialize the agent with tools
    # agent = initialize_agent(
    #     llm = llm,
    #     # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=False
    # )

    # # Run the agent on the formatted prompt
    # result = agent.run(prompt)

    result = llm.invoke(prompt).content.strip()

    # Extract final answer  

    return result.strip()