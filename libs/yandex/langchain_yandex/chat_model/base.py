from functools import cached_property
from typing import Optional

from langchain_core.load.serializable import Serializable

from .utils import MODEL2TYPE

from ..clients.foundation import FoundationModelClient, FoundationModel, TEMPERATURE, URL


class _BaseFoundationModel(Serializable):
    folder_id: str
    api_key: Optional[str] = None
    iam_token: Optional[str] = None
    model: FoundationModel = FoundationModel.YANDEXGPT_LITE
    base_url: str = URL
    temperature: float = TEMPERATURE
    max_tokens: Optional[int] = None
    streaming: bool = False
    reasoning: bool = False
    timeout: Optional[float] = None
    verbose: bool = False

    @property
    def _llm_type(self) -> str:
        return MODEL2TYPE[self.model]

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {
            "temperature": self.temperature,
            "model": self.model,
            "streaming": self.streaming,
            "reasoning": self.reasoning,
            "max_tokens": self.max_tokens
        }

    @cached_property
    def _client(self) -> FoundationModelClient:
        """Returns foundation model API client"""
        return FoundationModelClient(
            folder_id=self.folder_id,
            api_key=self.api_key,
            iam_token=self.iam_token,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            reasoning=self.reasoning,
            timeout=self.timeout
        )
