from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)


def convert_to_foundation_model_tool(tool: BaseTool) -> dict[str, dict[str, Any]]:
    return {
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema.model_json_schema()
        }
    }


def convert_to_foundation_model_message(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        yandexgpt_message = {"role": "system", "text": message.content}
    elif isinstance(message, HumanMessage):
        yandexgpt_message = {"role": "user", "text": message.content}
    elif isinstance(message, AIMessage):
        yandexgpt_message = {"role": "ai", "text": message.content}
        if hasattr(message, "tool_calls") and message.tool_calls:
            yandexgpt_message["toolCallList"] = {
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
        yandexgpt_message = {
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
    return yandexgpt_message
