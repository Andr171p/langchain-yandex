from typing import Any, Optional

from libs.yandex.langchain_yandex.chat_model.constants import YandexGPTModel, URL, ON_REASONING_MODE


class BaseYandexGPTClient:
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
