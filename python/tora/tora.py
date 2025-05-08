import httpx


def _format_hyperparams(hp: dict | None) -> list[dict]:
    formatted_hyperparams = []
    if hp:
        for key, value in hp.items():
            formatted_hyperparams.append({"key": key, "value": value})

    return formatted_hyperparams


class Tora:
    def __init__(
        self,
        experiment_id: str,
        server_url: str = "http://localhost:5173/api",
        max_buffer_len: int = 25,
    ):
        self._experiment_id = experiment_id
        self._max_buffer_len = max_buffer_len
        self._buffer = []
        self._http_client = httpx.Client(base_url=server_url)

    @classmethod
    def create_experiment(
        cls,
        name,
        description,
        hyperparams,
        tags,
        server_url: str = "http://localhost:5173/api",
    ):
        data = {"name": name}
        if description:
            data["description"] = description

        if hyperparams:
            data["hyperparams"] = _format_hyperparams(hyperparams)  # pyright: ignore

        if tags:
            data["tags"] = tags  # pyright: ignore

        req = httpx.post(
            server_url + "/experiments/create",
            json=data,
            headers={"Content-Type": "application/json"},
        )
        req.raise_for_status()
        return cls(req.json()["experiment"]["id"], server_url=server_url)

    def log(self, name, value, step: int | None = None, metadata: dict | None = None):
        self._buffer.append(
            {"name": name, "value": value, "step": step, "metadata": metadata}
        )
        if len(self._buffer) >= self._max_buffer_len:
            self._write_logs()

    def _write_logs(self):
        req = self._http_client.post(
            f"/experiments/{self._experiment_id}/metrics/batch",
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
