from typing import Any, Optional

import time

import requests

from .base import BaseYandexGPTClient
from .exceptions import CompletionError, BadRequest
from .constants import (
    OPERATIONS_ENDPOINT,
    ASYNC_TIMEOUT,
    STATUS_200_OK,
    STATUS_400_BAD_REQUEST,
    STATUS_500_INTERNAL_SERVER_ERROR
)


class YandexGPTClient(BaseYandexGPTClient):
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
        response = session.get(url=url, headers=headers)
        return response.json()
