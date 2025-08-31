from ib_insync import IB

class TwsConnection:
    """
    Thin wrapper around ib_insync.IB with context manager support.
    """
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 5
    ) -> None:
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

    def connect(self) -> IB:
        if not self.ib.isConnected():
            self.ib.connect(host=self.host, port=self.port, clientId=self.client_id, timeout=self.timeout)
        return self.ib

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def sleep(self, seconds: float) -> None:
        self.ib.sleep(seconds)

    def __enter__(self) -> "TwsConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()