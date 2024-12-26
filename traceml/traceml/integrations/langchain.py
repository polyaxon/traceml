import logging

from typing import Any, Optional

from polyaxon._client.decorators import client_handler
from polyaxon._schemas.lifecycle import V1StatusCondition, V1Statuses
from traceml import tracking
from traceml.events.schemas import V1EventSpan, V1EventSpanKind
from traceml.exceptions import TracemlException

try:
    from langchain.callbacks.tracers.base import BaseTracer
    from langchain.callbacks.tracers.schemas import Run, TracerSession
    from langchain.env import get_runtime_environment
    from langchain.load.dump import dumpd
    from langchain.schema.messages import BaseMessage
except ImportError:
    raise TracemlException("Langchain is required to use the tracking Callback")

logger = logging.getLogger(__name__)


def _serialize_io(run_inputs: dict) -> dict:
    serialized_inputs = {}
    for key, value in run_inputs.items():
        if key == "input_documents":
            serialized_inputs.update(
                {f"input_document_{i}": doc.to_json() for i, doc in enumerate(value)}
            )
        else:
            serialized_inputs[key] = value
    return serialized_inputs


class RunProcessor:
    """Handles the conversion of a LangChain Runs into a trace."""

    @classmethod
    def process_span(cls, run: Run) -> Optional["V1EventSpan"]:
        """Converts a LangChain Run into a V1EventSpan.
        Params:
            run: The LangChain Run to convert.
        Returns:
            The converted V1EventSpan.
        """
        try:
            span = cls._convert_lc_run_to_span(run)
            return span
        except Exception as e:
            logger.warning(
                f"Skipping trace saving - unable to safely convert LangChain Run "
                f"into Trace due to: {e}"
            )
            return None

    @classmethod
    def _convert_run_to_span(cls, run: Run) -> "V1EventSpan":
        """Base utility to create a span from a run.

        Params:
            run: The run to convert.
        Returns:
            The converted V1EventSpan.
        """
        metadata = {**run.extra} if run.extra else {}
        metadata["execution_order"] = run.execution_order

        status_conditions = (
            [
                V1StatusCondition.construct(
                    type=V1Statuses.FAILED,
                    status=True,
                    reason="SpanFailed",
                    message=run.error,
                    last_transition_time=run.end_time,
                    last_update_time=run.end_time,
                )
            ]
            if run.error
            else None
        )
        return V1EventSpan(
            uuid=str(run.id) if run.id is not None else None,
            name=run.name,
            started_at=run.start_time,
            finished_at=run.end_time,
            status=V1Statuses.SUCCEEDED if run.error is None else V1Statuses.FAILED,
            status_conditions=status_conditions,
            metadata=metadata,
        )

    @classmethod
    def _convert_llm_run_to_span(cls, run: Run) -> "V1EventSpan":
        """Converts a LangChain LLM Run into a V1EventSpan.
        Params
            run: The LangChain LLM Run to convert.
        Returns:
            The converted V1EventSpan.
        """
        base_span = cls._convert_run_to_span(run)
        if base_span.metadata is None:
            base_span.metadata = {}
        base_span.inputs = run.inputs
        base_span.outputs = run.outputs
        base_span.metadata["llm_output"] = run.outputs.get("llm_output", {})
        base_span.kind = V1EventSpanKind.LLM
        return base_span

    @classmethod
    def _convert_chain_run_to_span(cls, run: Run) -> "V1EventSpan":
        """Converts a LangChain Chain Run into a V1EventSpan.

        Params
            run: The LangChain Chain Run to convert.
        Returns:
            The converted V1EventSpan.
        """
        base_span = cls._convert_run_to_span(run)

        base_span.inputs = _serialize_io(run.inputs)
        base_span.outputs = _serialize_io(run.outputs)
        base_span.children = [
            cls._convert_lc_run_to_span(child_run) for child_run in run.child_runs
        ]
        base_span.kind = (
            V1EventSpanKind.AGENT
            if "agent" in run.name.lower()
            else V1EventSpanKind.CHAIN
        )

        return base_span

    @classmethod
    def _convert_tool_run_to_span(cls, run: Run) -> "V1EventSpan":
        """Converts a LangChain Tool Run into a V1EventSpan.

        Params
            run: The LangChain Tool Run to convert.
        Returns:
            The converted V1EventSpan.
        """
        base_span = cls._convert_run_to_span(run)
        base_span.inputs = _serialize_io(run.inputs)
        base_span.outputs = _serialize_io(run.outputs)
        base_span.children = [
            cls._convert_lc_run_to_span(child_run) for child_run in run.child_runs
        ]
        base_span.kind = V1EventSpanKind.TOOL

        return base_span

    @classmethod
    def _convert_lc_run_to_span(cls, run: Run) -> "V1EventSpan":
        """Utility to convert any generic LangChain Run into a V1EventSpan.

        Params
            run: The LangChain Run to convert.
        Returns:
            The converted V1EventSpan.
        """
        if run.run_type == V1EventSpanKind.LLM:
            return cls._convert_llm_run_to_span(run)
        elif run.run_type == V1EventSpanKind.CHAIN:
            return cls._convert_chain_run_to_span(run)
        elif run.run_type == V1EventSpanKind.TOOL:
            return cls._convert_tool_run_to_span(run)
        else:
            return cls._convert_run_to_span(run)


class Callback(BaseTracer):
    """Callback Handler that logs Langchain traces/spans."""

    @client_handler(check_no_op=True)
    def __init__(self, run=None, **kwargs: Any) -> None:
        """Initializes the callback.

        To monitor all LangChain activity, add this tracer like any other
        LangChain callback:

        ```python
        from polyaxon.tracking.integrations.langchain import Callback
        callback = Callback()
        chain = LLMChain(llm, callbacks=[callback])
        ```

        When using manual tracking of multiple runs in a single script:

        ```python
        from polyaxon.tracking.integrations.langchain import Callback
        tracking.init(..., is_new=True, ...)
        ...
        callback = Callback()
        chain = LLMChain(llm, callbacks=[callback])
        ...
        tracking.end()
        ```
        """
        super().__init__(**kwargs)
        self.run = tracking.get_or_create_run(run)

    @client_handler(check_no_op=True)
    def _persist_run(self, run: "Run"):
        """Converts a LangChain Run to a Trace."""
        span = RunProcessor.process_span(run)
        if span is None:
            return
        if self.run is not None:
            self.run.log_trace(span=span)
