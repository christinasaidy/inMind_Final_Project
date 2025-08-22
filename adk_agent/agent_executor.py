import logging
from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
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
from google.genai import types

if TYPE_CHECKING:
    from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_USER_ID = "self"


class OCRExecutor(AgentExecutor):

    """A2A AgentExecutor that runs  ADK  OCR agent """

    def __init__(self, runner: Runner, card: AgentCard):
        self.runner = runner
        self._card = card
        self._active_sessions: set[str] = set()

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> None:
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id
        self._active_sessions.add(session_id)

        try:
            async for event in self.runner.run_async(
                session_id=session_id,
                user_id=DEFAULT_USER_ID,
                new_message=new_message,
            ):
                if event.is_final_response():
                    parts = [
                        convert_genai_part_to_a2a(p)
                        for p in event.content.parts
                        if (p.text or p.file_data or p.inline_data)
                    ]
                    logger.debug("[availability] final response parts: %s", parts)
                    await task_updater.add_artifact(parts)
                    await task_updater.update_status(TaskState.completed, final=True)
                    break

                if not event.get_function_calls():
                    msg_parts = [
                        convert_genai_part_to_a2a(p)
                        for p in event.content.parts
                        if (p.text or p.file_data or p.inline_data)
                    ]
                    if msg_parts:
                        await task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message(msg_parts),
                        )
                else:
                    logger.debug("[availability] skipped function-call event")
        finally:
            self._active_sessions.discard(session_id)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.update_status(TaskState.submitted)
        await updater.update_status(TaskState.working)

        user_content = types.UserContent(
            parts=[convert_a2a_part_to_genai(p) for p in context.message.parts]
        )
        await self._process_request(
            new_message=user_content,
            session_id=context.context_id,
            task_updater=updater,
        )
        logger.debug("[availability] execute exiting")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        session_id = context.context_id
        if session_id in self._active_sessions:
            logger.info("Cancellation requested for active availability session: %s", session_id)
            self._active_sessions.discard(session_id)
        else:
            logger.debug("Cancellation requested for inactive availability session: %s", session_id)

        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str) -> "Session":
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
            )
        return session


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    root = part.root
    if isinstance(root, TextPart):
        return types.Part(text=root.text)
    if isinstance(root, FilePart):
        f = root.file
        if isinstance(f, FileWithUri):
            return types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type))
        if isinstance(f, FileWithBytes):
            raise ValueError("Inline bytes not supported. Upload to GCS and pass a file URI.")
        raise ValueError(f"Unsupported file type: {type(f)}")
    raise ValueError(f"Unsupported A2A part: {type(root)}")


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """
    Convert Google GenAI Part -> A2A Part (used for A2A updates/artifacts).
    """
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(file=FileWithUri(uri=part.file_data.file_uri, mime_type=part.file_data.mime_type))
    if part.inline_data:
        return FilePart(file=FileWithBytes(bytes=part.inline_data.data, mime_type=part.inline_data.mime_type))
    raise ValueError(f"Unsupported GenAI part: {part}")