import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LoanAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs ADK-based Loan Agent."""

    def __init__(self, runner: Runner):
        self.runner = runner
        self._running_sessions = {}

    def _run_agent(
        self, session_id, new_message: types.Content
    ) -> AsyncGenerator[Event, None]:
        return self.runner.run_async(
            session_id=session_id, user_id="loan_agent", new_message=new_message
        )

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> None:
        print(f"DEBUG: Processing request with message: {new_message}")
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id

        async for event in self._run_agent(session_id, new_message):
            if event.is_final_response():
                parts = convert_genai_parts_to_a2a(
                    event.content.parts if event.content and event.content.parts else []
                )
                print(f"DEBUG: Loan agent final response parts: {parts}")
                logger.debug("Yielding final response: %s", parts)
                task_updater.add_artifact(parts)
                task_updater.complete()
                break
            if not event.get_function_calls():
                logger.debug("Yielding update response")
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(
                            event.content.parts
                            if event.content and event.content.parts
                            else []
                        ),
                    ),
                )
            else:
                logger.debug("Skipping event")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        await self._process_request(
            types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id="loan_agent", session_id=session_id
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id="loan_agent",
                session_id=session_id,
            )
        if session is None:
            raise RuntimeError(f"Failed to get or create session: {session_id}")
        return session


def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert a list of A2A Part types into a list of Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type."""
    print(f"DEBUG: Converting A2A part to GenAI: {part}")
    root = part.root
    if isinstance(root, TextPart):
        print(f"DEBUG: Converting TextPart: {root.text}")
        return types.Part(text=root.text)
    if isinstance(root, FilePart):
        if isinstance(root.file, FileWithUri):
            print(f"DEBUG: Converting FileWithUri: {root.file.uri}")
            return types.Part(
                file_data=types.FileData(
                    file_uri=root.file.uri, mime_type=root.file.mimeType
                )
            )
        if isinstance(root.file, FileWithBytes):
            print(f"DEBUG: Converting FileWithBytes: {root.file.bytes[:100]}... (mimeType: {root.file.mimeType})")
            # For JSON content, we should pass it as text to the LLM, not as inline_data
            if root.file.mimeType == "application/json":
                print(f"DEBUG: Converting JSON to text: {root.file.bytes}")
                return types.Part(text=root.file.bytes)
            else:
                return types.Part(
                    inline_data=types.Blob(
                        data=root.file.bytes.encode("utf-8"),
                        mime_type=root.file.mimeType or "application/octet-stream",
                    )
                )
        raise ValueError(f"Unsupported file type: {type(root.file)}")
    raise ValueError(f"Unsupported part type: {type(part)}")


def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    """Convert a list of Google Gen AI Part types into a list of A2A Part types."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type."""
    if part.text:
        # Always try to parse as JSON first for loan agent responses
        text_content = part.text.strip()
        print(f"DEBUG: Converting GenAI text to A2A: {text_content[:100]}...")
        
        if text_content.startswith('{') and text_content.endswith('}'):
            try:
                # Validate it's valid JSON
                json.loads(text_content)
                print(f"DEBUG: Valid JSON detected, converting to FileWithBytes")
                # Convert to JSON blob with proper mime type
                return Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=text_content,
                            mimeType="application/json",
                        )
                    )
                )
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON decode error: {e}, keeping as text")
                # Not valid JSON, keep as text
                pass
        else:
            # If it doesn't look like JSON, wrap it in a JSON error response
            print(f"DEBUG: Non-JSON response detected, wrapping in error JSON")
            error_response = {
                "error": "invalid_response",
                "message": "Agent returned non-JSON response",
                "raw_response": text_content
            }
            return Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=json.dumps(error_response),
                        mimeType="application/json",
                    )
                )
            )
        # This should never happen for loan agent - all responses should be JSON
        # Wrap any remaining text in error JSON
        error_response = {
            "error": "unexpected_text_response",
            "message": "Loan agent returned unexpected text response",
            "raw_response": part.text
        }
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=json.dumps(error_response),
                    mimeType="application/json",
                )
            )
        )
    if part.file_data:
        if not part.file_data.file_uri:
            raise ValueError("File URI is missing")
        return Part(
            root=FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri,
                    mimeType=part.file_data.mime_type,
                )
            )
        )
    if part.inline_data:
        if not part.inline_data.data:
            raise ValueError("Inline data is missing")
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data.decode("utf-8"),
                    mimeType=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f"Unsupported part type: {part}")
