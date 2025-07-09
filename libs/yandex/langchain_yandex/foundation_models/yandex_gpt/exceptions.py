

class ClientError(Exception):
    pass


class CompletionError(ClientError):
    pass


class BadRequest(ClientError):
    pass
