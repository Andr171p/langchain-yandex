from typing import Any, Optional
from typing_extensions import TypedDict

import json
import logging

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool

from .base import _BaseFoundationModel
from .utils import (
    convert_dict_to_message,
    convert_message_to_dict,
    convert_tool_to_dict,
    create_chat_result
)

logger = logging.getLogger(__name__)


class Payload(TypedDict):
    messages: list[dict[str, str]]
    tools: Optional[list[dict[str, str]]]
    stop: Optional[list[str]]


class ChatFoundationModel(_BaseFoundationModel, BaseChatModel):

    def _build_payload(self, messages: list[BaseMessage], **kwargs: Any) -> Payload:
        message_dicts = [convert_message_to_dict(message) for message in messages]
        kwargs.pop("messages", None)
        tool_dicts = [
            convert_tool_to_dict(tool)
            for tool in kwargs.pop("tools", [])
            if isinstance(tool, BaseTool)
        ]
        stop = kwargs.pop("stop", None)
        payload = {
            "messages": message_dicts,
            "tools": tool_dicts if tool_dicts else None,
            "stop": stop
        }
        if self.verbose:
            logger.warning(
                "Foundation model request: %s",
                json.dumps(payload, ensure_ascii=False)
            )
        return payload

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        if self.iam_token:
            response = self._client.completion_async(**payload)
        else:
            response = self._client.completion(**payload)
        return create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        if self.iam_token:
            response = await self._client.acompletion_async(**payload)
        else:
            response = await self._client.acompletion(**payload)
        return create_chat_result(response)
