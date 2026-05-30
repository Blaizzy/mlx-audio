from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class InferenceResultChunk:
    kind: str
    payload: Any = None
    error: BaseException | None = None


@dataclass
class InferenceContext:
    request_id: str
    endpoint_kind: str
    model_name: str
    queued_at: float
    batch_key: Any = None


@dataclass
class InferenceRequest:
    endpoint_kind: str
    model_name: str
    payload: Any
    normalized_kwargs: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    batch_key: Any = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queued_at: float = field(default_factory=time.time)
    result_queue: "queue.Queue[InferenceResultChunk]" = field(
        default_factory=queue.Queue
    )
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def emit_data(self, payload: Any) -> None:
        self.result_queue.put(InferenceResultChunk(kind="data", payload=payload))

    def emit_error(self, error: BaseException) -> None:
        self.result_queue.put(InferenceResultChunk(kind="error", error=error))

    def emit_done(self) -> None:
        self.result_queue.put(InferenceResultChunk(kind="done"))


@dataclass
class InferenceHandle:
    context: InferenceContext
    result_queue: "queue.Queue[InferenceResultChunk]"
    cancel_event: threading.Event

    def cancel(self) -> None:
        self.cancel_event.set()


class ModelExecutionAdapter(Protocol):
    max_batch_size: int

    def supports_batch(self, request: InferenceRequest) -> bool: ...

    def batch_key(self, request: InferenceRequest) -> Any: ...

    def run_serial(self, request: InferenceRequest) -> None: ...

    def run_batch(self, requests: list[InferenceRequest]) -> None: ...

    def supports_continuous_batch(self, request: InferenceRequest) -> bool: ...

    def continuous_batch_key(self, request: InferenceRequest) -> Any: ...

    def create_continuous_batch_session(
        self, request: InferenceRequest
    ) -> "ContinuousBatchSession": ...


class ContinuousBatchSession(Protocol):
    @property
    def idle(self) -> bool: ...

    def submit(self, request: InferenceRequest) -> None: ...

    def step(self) -> None: ...

    def fail(self, error: BaseException) -> None: ...


class BaseModelExecutionAdapter:
    max_batch_size = 1

    def supports_batch(self, request: InferenceRequest) -> bool:
        del request
        return False

    def batch_key(self, request: InferenceRequest) -> Any:
        del request
        return None

    def run_serial(self, request: InferenceRequest) -> None:
        raise NotImplementedError

    def run_batch(self, requests: list[InferenceRequest]) -> None:
        if len(requests) != 1:
            raise NotImplementedError
        self.run_serial(requests[0])

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        del request
        return False

    def continuous_batch_key(self, request: InferenceRequest) -> Any:
        return self.batch_key(request)

    def create_continuous_batch_session(
        self, request: InferenceRequest
    ) -> ContinuousBatchSession:
        del request
        raise NotImplementedError


class InferenceBroker:
    def __init__(
        self,
        *,
        idle_poll_s: float = 0.1,
    ):
        self.idle_poll_s = idle_poll_s
        self._requests: "queue.Queue[Optional[InferenceRequest]]" = queue.Queue()
        self._adapters: dict[str, ModelExecutionAdapter] = {}
        self._continuous_sessions: dict[Any, ContinuousBatchSession] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register_adapter(
        self, endpoint_kind: str, adapter: ModelExecutionAdapter
    ) -> None:
        self._adapters[endpoint_kind] = adapter

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._requests.put(None)
        self._thread.join(timeout=timeout)
        for adapter in self._adapters.values():
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def submit(
        self,
        *,
        endpoint_kind: str,
        model_name: str,
        payload: Any,
        normalized_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        batch_key: Any = None,
    ) -> InferenceHandle:
        adapter = self._adapters.get(endpoint_kind)
        if adapter is None:
            raise ValueError(f"No inference adapter registered for {endpoint_kind!r}")

        request = InferenceRequest(
            endpoint_kind=endpoint_kind,
            model_name=model_name,
            payload=payload,
            normalized_kwargs=normalized_kwargs or {},
            stream=stream,
            batch_key=batch_key,
        )
        if request.batch_key is None:
            request.batch_key = adapter.batch_key(request)

        self._requests.put(request)
        return InferenceHandle(
            context=InferenceContext(
                request_id=request.request_id,
                endpoint_kind=request.endpoint_kind,
                model_name=request.model_name,
                queued_at=request.queued_at,
                batch_key=request.batch_key,
            ),
            result_queue=request.result_queue,
            cancel_event=request.cancel_event,
        )

    def _run(self) -> None:
        pending: list[InferenceRequest] = []
        try:
            while not self._stop.is_set():
                has_continuous_work = bool(self._continuous_sessions)
                self._fill_pending(
                    pending, block=not pending and not has_continuous_work
                )
                pending = [
                    request for request in pending if not request.cancel_event.is_set()
                ]

                self._route_continuous_requests(pending)
                self._step_continuous_sessions()

                # Keep all GPU work serialized. Whole-request serial or fixed-window
                # batch calls run only after active continuous sessions drain.
                if self._continuous_sessions:
                    continue

                if not pending:
                    continue

                request = pending.pop(0)
                adapter = self._adapters.get(request.endpoint_kind)
                if adapter is None:
                    request.emit_error(
                        ValueError(
                            f"No inference adapter registered for "
                            f"{request.endpoint_kind!r}"
                        )
                    )
                    request.emit_done()
                    continue

                requests = [request]
                if adapter.supports_batch(request) and adapter.max_batch_size > 1:
                    requests.extend(
                        self._select_batch_candidates(request, adapter, pending)
                    )

                try:
                    if len(requests) > 1:
                        adapter.run_batch(requests)
                    else:
                        adapter.run_serial(request)
                except Exception as exc:  # pragma: no cover - defensive broker guard
                    traceback.print_exc()
                    for failed_request in requests:
                        failed_request.emit_error(exc)
                        failed_request.emit_done()
        finally:
            for session in list(self._continuous_sessions.values()):
                session.fail(RuntimeError("Inference broker stopped."))
            self._continuous_sessions.clear()

    def _fill_pending(self, pending: list[InferenceRequest], *, block: bool) -> None:
        try:
            if block:
                item = self._requests.get(timeout=self.idle_poll_s)
            else:
                item = self._requests.get_nowait()
        except queue.Empty:
            return

        if item is None:
            self._stop.set()
            return
        pending.append(item)

        while True:
            try:
                item = self._requests.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._stop.set()
                return
            pending.append(item)

    def _select_batch_candidates(
        self,
        request: InferenceRequest,
        adapter: ModelExecutionAdapter,
        pending: list[InferenceRequest],
    ) -> list[InferenceRequest]:
        compatible: list[InferenceRequest] = []
        remaining: list[InferenceRequest] = []
        for candidate in pending:
            if len(compatible) >= adapter.max_batch_size - 1:
                remaining.append(candidate)
                continue
            if self._is_batch_compatible(request, candidate, adapter):
                compatible.append(candidate)
            else:
                remaining.append(candidate)
        pending[:] = remaining
        return compatible

    def _is_batch_compatible(
        self,
        request: InferenceRequest,
        candidate: InferenceRequest,
        adapter: ModelExecutionAdapter,
    ) -> bool:
        return (
            not candidate.cancel_event.is_set()
            and candidate.endpoint_kind == request.endpoint_kind
            and candidate.model_name == request.model_name
            and candidate.batch_key == request.batch_key
            and adapter.supports_batch(candidate)
        )

    def _route_continuous_requests(self, pending: list[InferenceRequest]) -> None:
        if not pending:
            return

        remaining: list[InferenceRequest] = []
        for request in pending:
            adapter = self._adapters.get(request.endpoint_kind)
            if adapter is None:
                remaining.append(request)
                continue
            if not adapter.supports_continuous_batch(request):
                remaining.append(request)
                continue

            session_key = (
                request.endpoint_kind,
                request.model_name,
                adapter.continuous_batch_key(request),
            )
            session = self._continuous_sessions.get(session_key)
            try:
                if session is None or session.idle:
                    session = adapter.create_continuous_batch_session(request)
                    self._continuous_sessions[session_key] = session
                session.submit(request)
            except Exception as exc:
                traceback.print_exc()
                request.emit_error(exc)
                request.emit_done()

        pending[:] = remaining

    def _step_continuous_sessions(self) -> None:
        for session_key, session in list(self._continuous_sessions.items()):
            try:
                session.step()
            except Exception as exc:
                traceback.print_exc()
                session.fail(exc)
                self._continuous_sessions.pop(session_key, None)
                continue

            if session.idle:
                self._continuous_sessions.pop(session_key, None)
