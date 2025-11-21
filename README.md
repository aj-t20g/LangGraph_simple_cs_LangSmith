# Product Support Agent

A customer support agent powered by LangGraph and Claude Sonnet 4 that provides product information, pricing, and inventory details.

## Features

- Product details and descriptions
- Pricing information
- SKU and inventory lookup
- Built with LangGraph for robust state management
- Powered by Claude Sonnet 4 (Anthropic) for intelligent responses
- LangSmith integration for monitoring and debugging

## Architecture

This agent uses LangGraph to create a stateful workflow:

```
START → Agent (Claude) ⟷ Tools → END
```

The agent can call three tools:
- `get_product_details`: Customer-facing product descriptions
- `get_product_price`: Product pricing
- `lookup_product_information`: Internal SKU and inventory data

## Prerequisites

- Python 3.11 or higher
- Anthropic API key
- (Optional) LangSmith API key for monitoring

## Local Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, for editable install (requires setuptools):
```bash
pip install setuptools wheel
pip install -e .
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=product-support-agent
```

### 3. Test Locally

Run the agent with the included test script:

```bash
python agent.py
```

This will run three test queries:
- "Tell me about the smartphone"
- "What is the price of headphones?"
- "What is the SKU for the speaker?"

### 4. Use in Your Code

```python
from agent import run_agent

# Simple invocation
response = run_agent("Tell me about the smartphone")
print(response)

# Or use the graph directly for more control
from agent import graph
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="What's the price of shoes?")]
})
print(result["messages"][-1].content)
```

## Deploying to LangGraph Cloud

### 1. Install LangGraph CLI

```bash
pip install langgraph-cli
```

### 2. Authenticate with LangSmith

```bash
# Set your LangSmith API key
export LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### 3. Create a LangGraph Cloud Deployment

First, ensure you have a LangGraph Cloud account at [smith.langchain.com](https://smith.langchain.com)

```bash
# Initialize the deployment (if not already done)
langgraph init

# Deploy to LangGraph Cloud
langgraph deploy
```

### 4. Set Environment Variables in LangGraph Cloud

After deployment, you need to configure environment variables in the LangGraph Cloud UI:

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your deployment
3. Go to Settings → Environment Variables
4. Add the following variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `LANGCHAIN_TRACING_V2`: `true`
   - `LANGCHAIN_API_KEY`: Your LangSmith API key
   - `LANGCHAIN_PROJECT`: `product-support-agent`

### 5. Test Your Deployment

Once deployed, you can invoke your agent via the LangGraph Cloud API or the LangSmith UI.

Example using the LangGraph SDK:

```python
from langgraph_sdk import get_client

client = get_client(url="your_deployment_url")

# Invoke the agent
response = client.runs.create(
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Tell me about the smartphone"}]},
)

print(response)
```

### 6. Monitor with LangSmith

View your agent's execution traces, debug issues, and monitor performance at [smith.langchain.com](https://smith.langchain.com)

## Configuration Files

- `langgraph.json`: LangGraph Cloud deployment configuration
- `pyproject.toml`: Python project dependencies
- `.env`: Environment variables (create from `.env.example`)

## Available Products

The agent has information about these products:
- smartphone
- usb charger
- shoes
- headphones
- speaker

## Development

### Project Structure

```
.
├── agent.py              # Main agent implementation
├── langgraph.json        # LangGraph Cloud config
├── pyproject.toml        # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Your environment variables (not in git)
└── README.md            # This file
```

### Adding New Products

To add new products, update the dictionaries in the tool functions in `agent.py`:

```python
@tool
def get_product_details(product_name: str) -> str:
    details = {
        "smartphone": "...",
        "new_product": "Product description here",  # Add your product
    }
    return details.get(product_name.lower(), "Product details not found.")
```

Make sure to update all three tool functions:
- `get_product_details`
- `get_product_price`
- `lookup_product_information`

## Troubleshooting

### API Key Issues

If you see authentication errors:
- Verify your `ANTHROPIC_API_KEY` is set correctly
- Check that your API key is active and has sufficient credits

### LangSmith Tracing Not Working

- Ensure `LANGCHAIN_TRACING_V2=true` is set
- Verify your `LANGCHAIN_API_KEY` is correct
- Check that your project name in `LANGCHAIN_PROJECT` exists

### Deployment Issues

- Run `langgraph doctor` to check your configuration
- Ensure `langgraph.json` points to the correct graph
- Verify all dependencies are listed in `pyproject.toml`

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/)

## License

This project is provided as-is for demonstration purposes.
