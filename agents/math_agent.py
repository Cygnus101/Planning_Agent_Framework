from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, load_tools, Tool, AgentType
from dotenv import load_dotenv
import os

load_dotenv()

# Load the math agent prompt template
with open("prompts/math_agent_prompt.txt", "r", encoding="utf-8") as f:
    math_agent_prompt_template = f.read()

def math_agent(task, history=None):
    # Format the prompt with task and history
    prompt = math_agent_prompt_template % (task, history if history else "None")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("API_KEY")
    )

    # Load calculator tool
    tools = load_tools(["llm-math"], llm=llm)

    # Initialize the agent with tools
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    # Run the agent on the formatted prompt
    result = agent.run(prompt)

    return result.strip()

# print(math_agent("What is the square root of 16?"))