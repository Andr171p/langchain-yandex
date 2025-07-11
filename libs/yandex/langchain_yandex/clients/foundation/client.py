from .sync_client import SyncFoundationModelClient
from .async_client import AsyncFoundationModelClient


class FoundationModelClient(SyncFoundationModelClient, AsyncFoundationModelClient):
    pass
