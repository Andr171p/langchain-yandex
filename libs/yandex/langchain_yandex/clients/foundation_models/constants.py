from enum import StrEnum


class FoundationModel(StrEnum):
    YANDEXGPT_LITE = "yandexgpt-lite"
    YANDEXGPT_PRO = "yandexgpt"


URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"
OPERATIONS_ENDPOINT = "https://operation.api.cloud.yandex.net/operations"

ASYNC_TIMEOUT = 1

TEMPERATURE = 0.7
ON_REASONING_MODE = "ENABLED_HIDDEN"

STATUS_200_OK = 200
STATUS_400_BAD_REQUEST = 400
STATUS_500_INTERNAL_SERVER_ERROR = 500
