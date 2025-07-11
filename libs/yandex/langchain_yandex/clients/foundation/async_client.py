from typing import Any, Optional

import asyncio

import aiohttp

from .base_client import BaseFoundationModelClient
from .exceptions import CompletionError, BadRequest
from .constants import (
    OPERATIONS_ENDPOINT,
    ASYNC_TIMEOUT,
    STATUS_200_OK,
    STATUS_400_BAD_REQUEST,
    STATUS_500_INTERNAL_SERVER_ERROR
)


class AsyncFoundationModelClient(BaseFoundationModelClient):
    async def acompletion(
            self,
            messages: list[dict[str, str]],
            tools: Optional[list[dict[str, Any]]] = None,
            stop: Optional[list[str]] = None
    ) -> dict[str, Any]:
        url = f"{self._base_url}/completion"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=url,
                    headers=self._headers,
                    json=self._build_payload(messages, tools, stop),
                    timeout=self._timeout
                ) as response:
                    status_code = response.status
                    if status_code == STATUS_200_OK:
                        return await response.json()
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
        except aiohttp.ClientError as e:
            raise CompletionError(f"Request failed: {e}") from e

    async def acompletion_async(
            self,
            messages: list[dict[str, str]],
            tools: Optional[list[str]] = None,
            stop: Optional[list[str]] = None,
            async_timeout: float = ASYNC_TIMEOUT
    ) -> dict[str, Any]:
        if not self._iam_token:
            raise ValueError("IAM-TOKEN required for this method")
        url = f"{self._base_url}/completionAsync"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=url,
                    headers=self._headers,
                    json=self._build_payload(messages, tools, stop)
                ) as response:
                    data = await response.json()
                    operation_id = data["id"]
                    while True:
                        status_operation = await self._aget_status_operation(session, operation_id)
                        await asyncio.sleep(async_timeout)
                        done = status_operation["done"]
                        if done:
                            return status_operation
        except aiohttp.ClientError as e:
            raise CompletionError(f"Request failed: {e}") from e

    async def _aget_status_operation(self, session: aiohttp.ClientSession, id: str) -> dict[str, Any]:
        url = f"{OPERATIONS_ENDPOINT}/{id}"
        headers = {"Authorization": f"Bearer {self._iam_token}"}
        async with session.get(url=url, headers=headers) as response:
            return response.json()
