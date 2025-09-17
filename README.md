# A2A 
This document describes a multi-agent application demonstrating how to orchestrate conversations between different agents.

This application contains four agents:
*   **Host Agent**: The primary agent that orchestrates the scheduling task.
*   **loan Agent**: An agent to approve or dispprove loans.

## Setup and Deployment

### Prerequisites

Before running the application locally, ensure you have the following installed:

1. **uv:** The Python package management tool used in this project. Follow the installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2. **python 3.13** Python 3.13 is required to run a2a-sdk 
3. **set up .env** 

Create a `.env` file in the root of the `a2a_friend_scheduling` directory with your Google API Key:
```
GOOGLE_API_KEY="your_api_key_here" 
```

## Run the Agents

You will need to run each agent in a separate terminal window. The first time you run these commands, `uv` will create a virtual environment and install all necessary dependencies before starting the agent.


### Terminal 3: Run loan Agent
```bash
cd loan_agent_adk
uv venv
source .venv/bin/activate
uv run --active .
```

### Terminal 4: Run Host Agent
```bash
cd host_agent_adk
uv venv
source .venv/bin/activate
uv run --active adk web      
```
