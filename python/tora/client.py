import httpx


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance


class Tora(Singleton):
    def __init__(
        self,
        name: str,
        description: str | None = None,
        hyperparams: dict[str, str | float | int] | None = None,
        tags: list[str] | None = None,
        max_buffer_len: int = 25,
    ):
        self._client = httpx.Client(base_url="http://localhost:5173/api")

        self._name = name
        self._description = description
        self._hyperparams = self._format_hyperparams(hyperparams)
        self._tags = tags
        self._id = self._create_experiment()
        self._max_buffer_len = max_buffer_len
        self._buffer = []

    def _format_hyperparams(self, hp: dict | None) -> list[dict]:
        formatted_hyperparams = []
        if hp:
            for key, value in hp.items():
                formatted_hyperparams.append({"key": key, "value": value})

        return formatted_hyperparams

    def _create_experiment(self):
        data = {"name": self._name}
        if self._description:
            data["description"] = self._description

        if self._hyperparams:
            data["hyperparams"] = self._hyperparams  # pyright: ignore

        if self._tags:
            data["tags"] = self._tags  # pyright: ignore

        req = self._client.post(
            "/experiments/create",
            json=data,
            headers={"Content-Type": "application/json"},
        )
        req.raise_for_status()
        return req.json()["experiment"]["id"]

    def log(self, name, value, step: int | None = None, metadata: dict | None = None):
        self._buffer.append(
            {"name": name, "value": value, "step": step, "metadata": metadata}
        )
        if len(self._buffer) >= self._max_buffer_len:
            self._write_logs()

    def _write_logs(self):
        req = self._client.post(
            f"/experiments/{self._id}/metrics/batch",
            headers={"Content-Type": "application/json"},
            json=self._buffer,
            timeout=120,
        )
        try:
            req.raise_for_status()
        except Exception as e:
            print(e, req.json())
        self._buffer = []

    def shutdown(self):
        if self._buffer:
            self._write_logs()
