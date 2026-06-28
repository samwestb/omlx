# SPDX-License-Identifier: Apache-2.0
"""Benchmark execution logic for oMLX admin panel.

Provides single-request and continuous-batching benchmarks with
real-time progress reporting via SSE events.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, field_validator

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Module-level storage for active benchmark runs
_benchmark_runs: dict[str, "BenchmarkRun"] = {}

# Valid prompt lengths for single request tests
VALID_PROMPT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]

# Valid batch sizes for continuous batching tests
VALID_BATCH_SIZES = [2, 4, 8]


class BenchmarkRequest(BaseModel):
    """Request model for starting a benchmark."""

    model_id: str
    prompt_lengths: list[int]
    generation_length: int = 128
    batch_sizes: list[int] = []
    force_lm_engine: bool = False

    @field_validator("prompt_lengths")
    @classmethod
    def validate_prompt_lengths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("At least one prompt length is required")
        for pl in v:
            if pl not in VALID_PROMPT_LENGTHS:
                raise ValueError(
                    f"Invalid prompt length {pl}. Must be one of {VALID_PROMPT_LENGTHS}"
                )
        return sorted(v)

    @field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, v: list[int]) -> list[int]:
        for bs in v:
            if bs not in VALID_BATCH_SIZES:
                raise ValueError(
                    f"Invalid batch size {bs}. Must be one of {VALID_BATCH_SIZES}"
                )
        return sorted(v)


@dataclass
class BenchmarkRun:
    """Tracks the state of a running benchmark.

    SSE delivery model: events are appended to `events` (append-only
    log) under `cond`. Subscribers replay `events` from offset 0 then
    wait on `cond` for new entries. `terminal` is set once the final
    event (`done` / `error`) has been published so subscribers
    know to close their stream rather than wait for a follow-up.
    """

    bench_id: str
    request: BenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: Optional[asyncio.Task] = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""


# Event types that close the SSE stream for a bench run.
_BENCH_TERMINAL_TYPES = frozenset({"done", "error"})


def get_run(bench_id: str) -> Optional[BenchmarkRun]:
    """Get a benchmark run by ID."""
    return _benchmark_runs.get(bench_id)


def get_active_run() -> Optional[BenchmarkRun]:
    """Return the currently-running throughput benchmark, if any.

    Discovery surface for clients that need to attach to an in-progress
    run without knowing the bench_id upfront (page refresh, second tab).
    Returns the first run with status == "running"; throughput benches
    are 1-at-a-time so there's never more than one.
    """
    for run in _benchmark_runs.values():
        if run.status == "running":
            return run
    return None


def create_run(request: BenchmarkRequest) -> BenchmarkRun:
    """Create and register a new benchmark run."""
    bench_id = f"bench-{uuid.uuid4().hex[:12]}"
    run = BenchmarkRun(bench_id=bench_id, request=request)
    _benchmark_runs[bench_id] = run
    return run


def cleanup_old_runs(max_runs: int = 10) -> None:
    """Remove old completed runs to prevent memory leaks."""
    completed = [
        (bid, r)
        for bid, r in _benchmark_runs.items()
        if r.status in ("completed", "cancelled", "error")
    ]
    if len(completed) > max_runs:
        for bid, _ in completed[:-max_runs]:
            del _benchmark_runs[bid]



def _generate_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Generate a prompt with exactly target_tokens tokens.

    Uses a unique UUID prefix to prevent SSD cache hits from previous sessions.
    """
    unique_prefix = f"BENCH-{uuid.uuid4().hex} "
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "In the realm of artificial intelligence, large language models "
        "have demonstrated remarkable capabilities across diverse tasks. "
    )

    # Build a large enough text
    text = unique_prefix + filler * (target_tokens // 10 + 1)
    tokens = tokenizer.encode(text)

    if len(tokens) < target_tokens:
        # Need more tokens, repeat more
        text = unique_prefix + filler * (target_tokens // 5 + 1)
        tokens = tokenizer.encode(text)

    # Truncate to exact target length
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def _compute_single_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float,
    end_time: float,
    peak_memory: int,
    cached_tokens: int,
    prefill_duration_s: float | None = None,
    generation_duration_s: float | None = None,
    generation_measured: bool = True,
) -> dict:
    """Compute all metrics for a single request benchmark."""
    ttft_s = first_token_time - start_time
    prefill_duration = (
        prefill_duration_s if prefill_duration_s is not None else ttft_s
    )
    gen_duration = (
        generation_duration_s
        if generation_duration_s is not None
        else end_time - first_token_time
    )
    e2e_duration = end_time - start_time

    ttft_ms = ttft_s * 1000
    if generation_measured and completion_tokens > 1 and gen_duration > 0:
        tpot_ms = (gen_duration / (completion_tokens - 1)) * 1000
        gen_tps = completion_tokens / gen_duration
    else:
        tpot_ms = 0.0
        gen_tps = 0.0
    processing_tps = prompt_tokens / max(prefill_duration, 1e-9)
    total_throughput = (prompt_tokens + completion_tokens) / max(e2e_duration, 1e-9)

    return {
        "ttft_ms": round(ttft_ms, 1),
        "tpot_ms": round(tpot_ms, 2),
        "gen_tps": round(gen_tps, 1),
        "processing_tps": round(processing_tps, 1),
        "e2e_latency_s": round(e2e_duration, 3),
        "total_throughput": round(total_throughput, 1),
        "peak_memory_bytes": peak_memory,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def _get_batch_benchmark_core(engine: Any) -> Any | None:
    """Return the scheduler core when this engine supports batch benchmarks."""
    engine_core = getattr(engine, "_engine", None)
    if engine_core is None:
        return None
    if not callable(getattr(engine_core, "add_request", None)):
        return None
    if not callable(getattr(engine_core, "stream_outputs", None)):
        return None
    return engine_core


async def _send_event(run: BenchmarkRun, event: dict) -> None:
    """Append an event to the run's log and wake any subscribers.

    Sets `run.terminal` when the event ends the stream so subscribers
    can return rather than wait for an event that will never come.
    """
    async with run.cond:
        run.events.append(event)
        if event.get("type") in _BENCH_TERMINAL_TYPES:
            run.terminal = True
        run.cond.notify_all()


async def _run_single_test(
    engine: Any,
    prompt: str,
    max_tokens: int,
    pp_len: int,
) -> dict:
    """Run a single request benchmark test and return metrics."""
    # Reset peak memory tracking
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    start_time = time.perf_counter()
    first_token_time = None
    last_generated_token_time = None
    last_output = None
    prev_completion_tokens = 0

    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    ):
        # Detect first generated token via completion_tokens count,
        # not new_text. Some models (e.g. Harmony/gpt-oss) produce
        # protocol tokens that don't yield visible new_text.
        completion_delta = output.completion_tokens - prev_completion_tokens
        if completion_delta > 0:
            generated_at = getattr(output, "generated_at", None)
            generated_until = getattr(output, "generated_until", None)
            output_first_token_time = (
                float(generated_at) if generated_at is not None else time.perf_counter()
            )
            if first_token_time is None:
                first_token_time = output_first_token_time
            if generated_until is not None:
                last_generated_token_time = float(generated_until)
            elif completion_delta == 1:
                last_generated_token_time = output_first_token_time
        prev_completion_tokens = output.completion_tokens
        last_output = output

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    # Get peak memory
    try:
        peak_memory = mx.get_peak_memory()
    except Exception:
        peak_memory = 0

    prompt_tokens = last_output.prompt_tokens if last_output else 0
    completion_tokens = last_output.completion_tokens if last_output else 0
    cached_tokens = last_output.cached_tokens if last_output else 0

    if cached_tokens > 0:
        logger.warning(
            f"Benchmark test pp{pp_len} had {cached_tokens} cached tokens "
            f"(expected 0). Results may not reflect true prefill performance."
        )

    prefill_duration_s = None
    generation_duration_s = None
    producer_generation_duration_s = None
    metric_completion_tokens = completion_tokens
    if first_token_time is not None and last_generated_token_time is not None:
        measured_duration = last_generated_token_time - first_token_time
        if measured_duration > 0:
            producer_generation_duration_s = measured_duration
    if last_output is not None:
        prompt_tps = float(getattr(last_output, "prompt_tps", 0.0) or 0.0)
        if prompt_tps > 0 and prompt_tokens > 0:
            prefill_duration_s = prompt_tokens / prompt_tps

        canvas_tps = float(getattr(last_output, "diffusion_canvas_tps", 0.0) or 0.0)
        canvas_tokens = int(getattr(last_output, "diffusion_canvas_tokens", 0) or 0)
        if canvas_tps > 0 and canvas_tokens > 0:
            metric_completion_tokens = canvas_tokens
            generation_duration_s = canvas_tokens / canvas_tps
        else:
            generation_tps = float(getattr(last_output, "generation_tps", 0.0) or 0.0)
            if generation_tps > 0 and completion_tokens > 0:
                generation_duration_s = completion_tokens / generation_tps

    if generation_duration_s is None:
        generation_duration_s = producer_generation_duration_s

    generation_measured = generation_duration_s is not None

    return _compute_single_metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=metric_completion_tokens,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        peak_memory=peak_memory,
        cached_tokens=cached_tokens,
        prefill_duration_s=prefill_duration_s,
        generation_duration_s=generation_duration_s,
        generation_measured=generation_measured,
    )


async def _run_batch_test(
    engine: Any,
    prompts: list[str],
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Run a continuous batching benchmark test.

    Submits batch_size concurrent requests via the engine core and measures
    aggregate throughput including pp TPS and tg TPS.

    Args:
        prompts: List of prompts (one per request). For same-prompt tests,
                 all entries are identical. For different-prompt tests, each
                 has a unique UUID prefix.
        prompt_tokens: Number of prompt tokens per request (for pp TPS calc).
    """
    from ..request import SamplingParams

    engine_core = _get_batch_benchmark_core(engine)
    if engine_core is None:
        raise ValueError("Engine does not support batch benchmarks")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    async def _single_request(prompt: str) -> dict:
        """Run a single request within the batch."""
        start = time.perf_counter()
        first_token = None
        tokens = 0
        prev_tokens = 0

        request_id = await engine_core.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        async for output in engine_core.stream_outputs(request_id):
            if first_token is None and output.completion_tokens > prev_tokens:
                first_token = time.perf_counter()
            prev_tokens = output.completion_tokens
            if output.finished:
                tokens = output.completion_tokens

        end = time.perf_counter()
        if first_token is None:
            first_token = end

        return {
            "ttft_s": first_token - start,
            "first_token_abs": first_token,
            "end_abs": end,
            "completion_tokens": tokens,
        }

    # Submit all requests concurrently
    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_single_request(prompts[i]) for i in range(batch_size)]
    )
    wall_end = time.perf_counter()

    # Aggregate metrics
    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = prompt_tokens * batch_size
    wall_time = wall_end - wall_start
    avg_ttft_ms = (sum(r["ttft_s"] for r in results) / batch_size) * 1000

    # pp TPS: total prompt tokens / time until ALL requests finish prefill
    max_first_token = max(r["first_token_abs"] for r in results)
    prefill_wall_time = max_first_token - wall_start
    pp_tps = total_prompt_tokens / max(prefill_wall_time, 1e-9)

    # tg TPS: total generated tokens / generation wall time
    # Generation starts when the last request finishes prefill
    gen_wall_time = wall_end - max_first_token
    tg_tps = total_gen_tokens / max(gen_wall_time, 1e-9)

    return {
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "e2e_latency_s": round(wall_time, 3),
        "total_gen_tokens": total_gen_tokens,
        "batch_size": batch_size,
    }


async def run_benchmark(run: BenchmarkRun, engine_pool: Any) -> None:
    """Execute a complete benchmark run.

    Phases:
    1. Unload all loaded models
    2. Load the target model
    3. Run single request tests
    4. Run batch tests
    5. Unload the benchmark model
    """
    request = run.request
    total_tests = len(request.prompt_lengths) + len(request.batch_sizes)
    current_test = 0
    overall_start = time.perf_counter()

    try:
        # Read model settings so VLM MTP can pick the right engine below.
        model_settings = None
        sm = getattr(engine_pool, "_settings_manager", None)
        if sm is not None:
            try:
                model_settings = sm.get_settings(request.model_id)
            except Exception as e:
                logger.warning(
                    f"Benchmark: failed to read settings for "
                    f"{request.model_id}: {e}"
                )

        # Phase 1: Unload all loaded models
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _send_event(run, {
                "type": "progress",
                "phase": "unload",
                "message": f"Unloading {len(loaded_ids)} model(s)...",
                "current": 0,
                "total": total_tests,
            })
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                    logger.info(f"Benchmark: unloaded {model_id}")
                except Exception as e:
                    logger.warning(f"Benchmark: failed to unload {model_id}: {e}")

        # Phase 2: Load the target model
        await _send_event(run, {
            "type": "progress",
            "phase": "load",
            "message": f"Loading {request.model_id}...",
            "current": 0,
            "total": total_tests,
        })
        # VLM MTP requires VLMBatchedEngine (which has set_vlm_mtp_drafter),
        # so don't force LM-only loading when VLM MTP is enabled.
        vlm_mtp_active = (
            model_settings is not None
            and getattr(model_settings, "vlm_mtp_enabled", False)
            and getattr(model_settings, "vlm_mtp_draft_model", None)
        )
        force_lm = True if request.force_lm_engine else not vlm_mtp_active
        engine = await engine_pool.get_engine(
            request.model_id,
            force_lm=force_lm,
        )
        logger.info(f"Benchmark: loaded {request.model_id}")

        # Generate prompts for all needed lengths
        tokenizer = engine.tokenizer
        prompts: dict[int, str] = {}
        for pp_len in request.prompt_lengths:
            prompts[pp_len] = _generate_prompt(tokenizer, pp_len)

        # Ensure pp1024 prompt exists for batch tests
        if request.batch_sizes and 1024 not in prompts:
            prompts[1024] = _generate_prompt(tokenizer, 1024)

        # Warmup: run a short request to trigger JIT compilation,
        # Metal shader compilation, and KV cache initialization.
        # Without this, the first real benchmark test absorbs all
        # one-time overhead and shows artificially low pp TPS.
        await _send_event(run, {
            "type": "progress",
            "phase": "warmup",
            "message": "Warming up (JIT compile)...",
            "current": 0,
            "total": total_tests,
        })
        warmup_prompt = _generate_prompt(tokenizer, 32)
        warmup_max_tokens = (
            request.generation_length
            if getattr(engine, "is_diffusion_model", False)
            else 8
        )
        async for _ in engine.stream_generate(
            prompt=warmup_prompt, max_tokens=warmup_max_tokens, temperature=0.0
        ):
            pass
        logger.info("Benchmark: warmup complete")

        # Phase 3: Single request tests
        single_pp1024_gen_tps = None

        for pp_len in request.prompt_lengths:
            current_test += 1
            await _send_event(run, {
                "type": "progress",
                "phase": "single",
                "message": f"Single: pp{pp_len}/tg{request.generation_length}",
                "current": current_test,
                "total": total_tests,
            })

            metrics = await _run_single_test(
                engine=engine,
                prompt=prompts[pp_len],
                max_tokens=request.generation_length,
                pp_len=pp_len,
            )

            result = {
                "test_type": "single",
                "pp": pp_len,
                "tg": request.generation_length,
                **metrics,
            }
            run.results.append(result)

            await _send_event(run, {"type": "result", "data": result})

            # Store pp1024 gen_tps for speedup calculation
            if pp_len == 1024:
                single_pp1024_gen_tps = metrics["gen_tps"]

        # Phase 4: Batch tests
        # Each request has a unique UUID prefix (no cache hits)
        max_batch = max(request.batch_sizes) if request.batch_sizes else 0
        batch_prompts = [_generate_prompt(tokenizer, 1024) for _ in range(max_batch)]

        # Skip batch tests for engines without scheduler core (e.g. VLM/Diffusion)
        batch_core = _get_batch_benchmark_core(engine)
        if request.batch_sizes and batch_core is None:
            logger.info(
                "Batch test skipped: engine does not support concurrent batching"
            )
            current_test += len(request.batch_sizes)

        for batch_size in request.batch_sizes if batch_core is not None else []:
            current_test += 1
            await _send_event(run, {
                "type": "progress",
                "phase": "batch",
                "message": f"Batch {batch_size}x: pp1024/tg{request.generation_length}",
                "current": current_test,
                "total": total_tests,
            })

            batch_metrics = await _run_batch_test(
                engine=engine,
                prompts=batch_prompts[:batch_size],
                prompt_tokens=1024,
                max_tokens=request.generation_length,
                batch_size=batch_size,
            )

            result = {
                "test_type": "batch",
                "pp": 1024,
                "tg": request.generation_length,
                **batch_metrics,
            }
            run.results.append(result)
            await _send_event(run, {"type": "result", "data": result})

        # Phase 5: Unload benchmark model
        await _send_event(run, {
            "type": "progress",
            "phase": "cleanup",
            "message": f"Unloading {request.model_id}...",
            "current": total_tests,
            "total": total_tests,
        })
        try:
            await engine_pool._unload_engine(request.model_id)
            logger.info(f"Benchmark: unloaded {request.model_id} after benchmark")
        except Exception as e:
            logger.warning(f"Benchmark: failed to unload {request.model_id}: {e}")

        # Done
        overall_duration = time.perf_counter() - overall_start
        run.status = "completed"
        await _send_event(run, {
            "type": "done",
            "summary": {
                "model_id": request.model_id,
                "total_time": round(overall_duration, 1),
                "total_tests": total_tests,
            },
        })

    except asyncio.CancelledError:
        run.status = "cancelled"
        await _send_event(run, {
            "type": "error",
            "message": "Benchmark cancelled by user",
        })
        # Try to unload the model on cancellation
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        run.status = "error"
        run.error_message = str(e)
        await _send_event(run, {
            "type": "error",
            "message": str(e),
        })
        # Try to unload the model on error
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass
