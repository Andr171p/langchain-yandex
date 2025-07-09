from typing import Any, Optional

import time

import requests

from .constants import (
    YandexGPTModel,
    URL,
    OPERATIONS_ENDPOINT,
    ON_REASONING_MODE,
    ASYNC_TIMEOUT,
    STATUS_200_OK,
    STATUS_400_BAD_REQUEST,
    STATUS_500_INTERNAL_SERVER_ERROR
)
from .exceptions import CompletionError, BadRequest, OperationError


class YandexGPTClient:
    def __init__(
            self,
            folder_id: str,
            api_key: Optional[str] = None,
            iam_key: Optional[str] = None,
            model: YandexGPTModel = YandexGPTModel.LITE,
            base_url: str = URL,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            streaming: bool = False,
            reasoning: bool = False,
            timeout: Optional[float] = None
    ) -> None:
        self._folder_id = folder_id
        """ Identifier of cloud catalog with current role ai.languageModels.user or higher """
        self._api_key = api_key
        """ API key for send request """
        self._iam_token = iam_key
        """ Authorization token """
        self._model = model
        """ Model name to use """
        self._base_url = base_url
        """ Base API URL """
        self._temperature = temperature
        """ What sampling temperature to use """
        self._max_tokens = max_tokens
        """ Maximum number of tokens to generate """
        self._streaming = streaming
        """ Whether to stream the results or not """
        self._reasoning = reasoning
        """ Whether to reasoning or not"""
        self._timeout = timeout
        """ Timeout for request """

    @property
    def _model_uri(self) -> str:
        return f"gpt://{self._folder_id}/{self._model}"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-folder-id": self._folder_id
        }
        if self._api_key:
            headers["Authorization"] = f"Api-Key {self._api_key}"
        elif self._iam_token:
            headers["Authorization"] = f"Bearer {self._iam_token}"
        else:
            raise ValueError("IAM-TOKEN or API-KEY is not set")
        return headers

    def _build_payload(
            self,
            messages: list[dict[str, str]],
            tools: Optional[list[dict[str, Any]]] = None,
            stop: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Method for build JSON data.

        :param messages: Messages in friendly YandexGPT format
        :param tools: Tools for function calling
        :param stop: Sequence of words to stop generation
        :return: Built JSON for sending requests
        """
        payload = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": self._streaming,
                "temperature": self._temperature,
                "maxTokens": self._max_tokens,
            },
            "messages": messages
        }
        if self._reasoning:
            payload["reasoningOptions"]["mode"] = ON_REASONING_MODE
        if tools:
            payload["tools"] = tools
        if stop:
            payload["completionOptions"]["stopSequences"] = stop
        return payload

    def completion(
            self,
            messages: list[dict[str, str]],
            tools: Optional[list[dict[str, Any]]] = None,
            stop: Optional[list[str]] = None
    ) -> dict[str, Any]:
        url = f"{self._base_url}/completion"
        try:
            with requests.Session() as session:
                response = session.post(
                    url=url,
                    headers=self._headers,
                    json=self._build_payload(messages, tools, stop),
                    timeout=self._timeout
                )
            status_code = response.status_code
            if status_code == STATUS_200_OK:
                return response.json()
            elif STATUS_400_BAD_REQUEST <= status_code < STATUS_500_INTERNAL_SERVER_ERROR:
                raise BadRequest(
                    f"Bad request (status {status_code}):"
                    f"{response.text}"
                )
            else:
                raise CompletionError(
                    f"Server error (status {status_code}):"
                    f"{response.text}"
                )
        except requests.RequestException as e:
            raise CompletionError(f"Request failed: {e}") from e

    def completion_async(
            self,
            messages: list[dict[str, str]],
            tools: Optional[list[dict[str, Any]]] = None,
            stop: Optional[list[str]] = None,
            async_timeout: float = ASYNC_TIMEOUT
    ) -> dict[str, Any]:
        if not self._iam_token:
            raise ValueError("IAM-TOKEN required for this method")
        url = f"{self._base_url}/completionAsync"
        try:
            with requests.Session() as session:
                response = session.post(
                    url=url,
                    headers=self._headers,
                    json=self._build_payload(messages, tools, stop)
                    )
                data = response.json()
                operation_id = data["id"]
                while True:
                    status_operation = self._get_status_operation(session, operation_id)
                    time.sleep(async_timeout)
                    done = status_operation["done"]
                    if done:
                        return status_operation
        except requests.RequestException as e:
            raise CompletionError(f"Request failed: {e}") from e

    def _get_status_operation(self, session: requests.Session, id: str) -> dict[str, Any]:
        url = f"{OPERATIONS_ENDPOINT}/{id}"
        headers = {"Authorization": f"Bearer {self._iam_token}"}
        try:
            response = session.get(url=url, headers=headers)
            return response.json()
        except requests.RequestException as e:
            raise OperationError(f"Request failed: {e}") from e
