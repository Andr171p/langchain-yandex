from typing import Any

from uuid import uuid4
from enum import StrEnum

from langchain_core.tools import BaseTool
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)

from ..clients.foundation import FoundationModel

YANDEXGPT_TYPE = "YandexGPT"

MODEL2TYPE: dict[FoundationModel, str] = {
    FoundationModel.YANDEXGPT_LITE: YANDEXGPT_TYPE,
    FoundationModel.YANDEXGPT_PRO: YANDEXGPT_TYPE
}

TOOL_CALLS_STATUS = "ALTERNATIVE_STATUS_TOOL_CALLS"


class FoundationMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def convert_tool_to_dict(tool: BaseTool) -> dict[str, dict[str, Any]]:
    return {
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema.model_json_schema()
        }
    }


def convert_message_to_dict(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        message_dict = {"role": "system", "text": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "text": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "ai", "text": message.content}
        if hasattr(message, "tool_calls") and message.tool_calls:
            message_dict["toolCallList"] = {
                "toolCalls": [
                    {
                        "functionCall": {
                            "name": tool_call["name"],
                            "argument": tool_call["args"]
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {
                        "functionResult": {
                            "name": message.name,
                            "content": message.content
                        }
                    }
                ]
            }
        }
    else:
        raise TypeError("This message type not supported by YandexGPT")
    return message_dict


def convert_dict_to_message(message: dict[str, Any]) -> BaseMessage:
    additional_kwargs = {}
    tool_calls: list[ToolCall] = []
    if hasattr(message, "toolCallList"):
        called_tools = message["toolCallList"]["toolCalls"]
        additional_kwargs["toolCals"] = []
        for called_tool in called_tools:
            additional_kwargs["toolCals"].append(called_tool["functionCall"])
    if additional_kwargs.get("toolCalls"):
        tool_calls = [
            ToolCall(
                name=tool_call["name"],
                args=tool_call["arguments"],
                id=str(uuid4())
            )
            for tool_call in additional_kwargs["tollCalls"]
        ]
    if hasattr(message, "toolResultList"):
        tool_results = message["toolResultList"]["toolResults"]
        return ToolMessage(
            content=tool_results[0]["content"],
            name=tool_results[0]["name"],
            tool_call_id=str(uuid4())
        )
    role = message["role"]
    content = message["text"]
    if role == FoundationMessageRole.SYSTEM:
        return SystemMessage(content=content)
    elif role == FoundationMessageRole.USER:
        return HumanMessage(content=content)
    elif role == FoundationMessageRole.ASSISTANT:
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls
        )
    else:
        raise TypeError(f"Got unknown role {role} {message}")


def create_chat_result(response: dict[str, Any]) -> ChatResult:
    generations: list[ChatGeneration] = []
    alternatives: list[dict[str, Any]] = response["result"]["alternatives"]
    for alternative in alternatives:
        message = convert_dict_to_message(alternative["message"])
        if isinstance(message, AIMessage):
            message.usage_metadata = UsageMetadata(
                output_tokens=response["usage"]["completionTokens"],
                input_tokens=response["usage"]["inputTextTokens"],
                total_tokens=response["usage"]["totalTokens"],
                input_token_details={
                    "reasoning_tokens": response["usage"]["completionTokensDetails"]["reasoningTokens"]
                }
            )
        generation = ChatGeneration(
            message=message,
            generation_info={"model_version": response["modelVersion"]}
        )
        generations.append(generation)
    return ChatResult(generations=generations)
