"""Product Support Agent using LangGraph and Claude."""

from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# Define the agent state
class AgentState(TypedDict):
    """State for the product support agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# Tool Definitions
@tool
def get_product_details(product_name: str) -> str:
    """Gathers details about a product in the catalog.

    Args:
        product_name: The name of the product (e.g., smartphone, usb charger, shoes, headphones, speaker)

    Returns:
        Product description and details
    """
    details = {
        "smartphone": "A cutting-edge smartphone with advanced camera features and lightning-fast processing.",
        "usb charger": "A super fast and light usb charger",
        "shoes": "High-performance running shoes designed for comfort, support, and speed.",
        "headphones": "Wireless headphones with advanced noise cancellation technology for immersive audio.",
        "speaker": "A voice-controlled smart speaker that plays music, sets alarms, and controls smart home devices.",
    }
    return details.get(product_name.lower(), "Product details not found.")


@tool
def get_product_price(product_name: str) -> str:
    """Gathers price about a product.

    Args:
        product_name: The name of the product

    Returns:
        Price as a string
    """
    prices = {
        "smartphone": "500",
        "usb charger": "10",
        "shoes": "100",
        "headphones": "50",
        "speaker": "80",
    }
    price = prices.get(product_name.lower())
    return f"${price}" if price else "Price not found."


@tool
def lookup_product_information(product_name: str) -> str:
    """Looks up specific information for a product in the catalog.

    Args:
        product_name: The name of the product

    Returns:
        Backend information including SKU and inventory
    """
    backend_info = {
        "smartphone": "SKU: G-SMRT-001, Inventory: 550 units",
        "usb charger": "SKU: G-CHRG-003, Inventory: 1200 units",
        "shoes": "SKU: G-SHOE-007, Inventory: 800 units",
        "headphones": "SKU: G-HDPN-002, Inventory: 950 units",
        "speaker": "SKU: G-SPKR-001, Inventory: 400 units",
    }
    return backend_info.get(product_name.lower(), "Backend information not found.")


# Collect all tools
tools = [get_product_details, get_product_price, lookup_product_information]

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful customer support agent specializing in product information.
Your goal is to answer user queries about product details or prices.

1. For general, customer-facing descriptions (like 'tell me about...'), ALWAYS use the `get_product_details` tool.
2. For internal data like SKU or inventory, use the `lookup_product_information` tool.
3. If the user is asking for the price, use the `get_product_price` tool.

Available products: smartphone, usb charger, shoes, headphones, speaker
"""


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue or end the conversation.

    Args:
        state: The current agent state

    Returns:
        "tools" if there are tool calls to execute, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then we route to the "tools" node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "end"


def call_model(state: AgentState) -> AgentState:
    """Call the LLM model with the current state.

    Args:
        state: The current agent state

    Returns:
        Updated state with the model's response
    """
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Initialize the model
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
    )
    model_with_tools = model.bind_tools(tools)

    # Call the model
    response = model_with_tools.invoke(messages)

    # Return the updated state
    return {"messages": [response]}


# Build the graph
def create_graph():
    """Create and compile the LangGraph workflow.

    Returns:
        Compiled LangGraph workflow
    """
    import os

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entry point
    workflow.add_edge(START, "agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    # LangSmith tracing is automatically enabled when LANGSMITH_API_KEY is set
    # Set LANGCHAIN_TRACING_V2=true to enable tracing
    return workflow.compile()


# Create the compiled graph (the agent)
graph = create_graph()


# Helper function for easy invocation
def run_agent(user_input: str) -> str:
    """Run the agent with a user input.

    Args:
        user_input: The user's question or request

    Returns:
        The agent's response
    """
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
    )
    return result["messages"][-1].content


# Example usage (for testing)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set it in a .env file or as an environment variable")
        exit(1)

    # Test the agent
    print("Product Support Agent (powered by Claude and LangGraph)")
    print("=" * 60)

    test_queries = [
        "Tell me about the smartphone",
        "What is the price of headphones?",
        "What is the SKU for the speaker?",
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        response = run_agent(query)
        print(f"Agent: {response}")
