from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Now we look in tools for the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# First search tool
search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_datetime(formate:str = "%Y-%m-%d %H:%M:%S") -> datetime.datetime:
    """Get the current system date and time."""
    current_time =datetime.datetime.now()
    return current_time.strftime(formate)
tools = [search_tool, get_system_datetime]
# Initialize the agent with the LLM and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)
agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")
