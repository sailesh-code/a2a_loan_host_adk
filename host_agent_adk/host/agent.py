import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .host_tools import (
    make_loan_application,
)
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class HostAgent:
    """The Host agent."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash",
            name="Host_Agent",
            instruction=self.root_instruction,
            description="This Host agent orchestrates loan applications to bank agents.",
            tools=[
                self.send_message,
                make_loan_application,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:** You are the Host Agent, an expert coordinator for loan applications to bank agents. Your primary function is to coordinate with bank agents to process loan applications using JSON communication.

        **Core Directives:**
        
        **Loan Application Workflow:**
        1. When asked to make a loan application, first use `make_loan_application(loan_id, amount)` 
        2. Take the `json_payload` from `make_loan_application` output and pass it directly to `send_message`
        3. Use `send_message(agent_name, json_payload, tool_context)` to send the JSON data to the bank agent
        
        **Example Workflow:**
        - User: "Make a loan application for loan_id 12345 with amount 50000"
        - Step 1: Call `make_loan_application("12345", "50000")` → returns {{"json_payload": {{"type": "loan_status_request", "loan_id": "12345", "amount": 50000.0}}, "message": "..."}}
        - Step 2: Call `send_message("Loan Agent", result["json_payload"], tool_context)`
        
        **Important:** 
        - Always use both tools in sequence - first `make_loan_application`, then `send_message`
        - Pass the `json_payload` directly to `send_message` - no text conversion needed
        - All communication is now pure JSON format
        
        <Available Agents>
        {self.agents}
        </Available Agents>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        # Convert user query to JSON format for consistency
        query_json = {
            "type": "user_query",
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        content = types.Content(role="user", parts=[types.Part.from_text(text=json.dumps(query_json))])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    # Try to parse as JSON first
                    text_content = event.content.parts[0].text.strip()
                    try:
                        # If it's valid JSON, use it directly
                        json.loads(text_content)
                        response = text_content
                    except json.JSONDecodeError:
                        # If not JSON, wrap in JSON format
                        response = json.dumps({
                            "type": "text_response",
                            "content": text_content,
                            "timestamp": datetime.now().isoformat()
                        })
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "The host agent is thinking...",
                }

    async def send_message(self, agent_name: str, json_data: dict, tool_context: ToolContext):
        """Sends JSON data to a remote friend agent."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        # Simplified task and context ID management
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        print(f"DEBUG: send_message called with JSON data: {json_data}")
        
        # Always send as JSON - no text fallback needed
        payload = {
            "message": {
                "role": "user",
                "parts": [{
                    "type": "file",
                    "file": {
                        "bytes": json.dumps(json_data),
                        "mimeType": "application/json"
                    }
                }],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)
        
        print(f"DEBUG: Response structure: {json_content.get('result', {}).keys()}")

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            print(f"DEBUG: Found {len(json_content['result']['artifacts'])} artifacts")
            for i, artifact in enumerate(json_content["result"]["artifacts"]):
                if artifact.get("parts"):
                    print(f"DEBUG: Found {len(artifact['parts'])} parts in artifact {i}")
                    for j, part in enumerate(artifact["parts"]):
                        print(f"DEBUG: Part {j}: type={part.get('type')}, has_file={bool(part.get('file'))}")
                        # Handle JSON responses with bytes (check for file with JSON mimeType)
                        if (part.get("file", {}).get("mimeType") == "application/json"):
                            print(f"DEBUG: Processing JSON file part (type={part.get('type')})")
                            try:
                                json_data = json.loads(part["file"]["bytes"])
                                print(f"DEBUG: Successfully parsed JSON: {json_data}")
                                resp.append(json_data)
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"DEBUG: JSON decode error: {e}")
                                resp.append(part)
                        # Handle text parts that might contain JSON
                        elif part.get("type") == "text":
                            print(f"DEBUG: Processing text part: {part.get('text', '')[:100]}...")
                            try:
                                # Try to parse as JSON
                                json_data = json.loads(part["text"])
                                print(f"DEBUG: Successfully parsed text as JSON: {json_data}")
                                resp.append(json_data)
                            except (json.JSONDecodeError, KeyError):
                                # Wrap non-JSON text in error format
                                error_response = {
                                    "error": "text_response",
                                    "message": "Received text response instead of JSON",
                                    "text_content": part.get("text", "")
                                }
                                resp.append(error_response)
                        # Handle any remaining parts (should all be JSON now)
                        else:
                            print(f"DEBUG: Unexpected part type: {part.get('type')}")
                            # If we get unexpected part types, wrap in error JSON
                            error_response = {
                                "error": "unexpected_response_format",
                                "message": "Received unexpected response format",
                                "part_type": part.get("type", "unknown"),
                                "part_keys": list(part.keys()),
                                "full_part": part
                            }
                            resp.append(error_response)
        else:
            print(f"DEBUG: No artifacts found in response")
            error_response = {
                "error": "no_artifacts",
                "message": "No artifacts found in response",
                "response_structure": json_content
            }
            resp.append(error_response)
        return resp


def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # Hardcoded URLs for the friend agents
        friend_agent_urls = [
            "http://localhost:10002",  # Loan Agent
        ]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync()
