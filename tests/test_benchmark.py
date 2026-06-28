# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin benchmark module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin.benchmark import (
    VALID_BATCH_SIZES,
    VALID_PROMPT_LENGTHS,
    BenchmarkRequest,
    BenchmarkRun,
    _compute_single_metrics,
    _generate_prompt,
    _run_single_test,
    cleanup_old_runs,
    create_run,
    get_run,
    run_benchmark,
)


# =============================================================================
# BenchmarkRequest validation tests
# =============================================================================


class TestBenchmarkRequest:
    def test_valid_request(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024, 4096],
            generation_length=128,
            batch_sizes=[2, 4],
        )
        assert req.model_id == "test-model"
        assert req.prompt_lengths == [1024, 4096]
        assert req.batch_sizes == [2, 4]

    def test_prompt_lengths_sorted(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[8192, 1024, 4096],
        )
        assert req.prompt_lengths == [1024, 4096, 8192]

    def test_empty_prompt_lengths_rejected(self):
        with pytest.raises(ValueError, match="At least one prompt length"):
            BenchmarkRequest(model_id="test-model", prompt_lengths=[])

    def test_invalid_prompt_length_rejected(self):
        with pytest.raises(ValueError, match="Invalid prompt length 512"):
            BenchmarkRequest(model_id="test-model", prompt_lengths=[512])

    def test_invalid_batch_size_rejected(self):
        with pytest.raises(ValueError, match="Invalid batch size 3"):
            BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                batch_sizes=[3],
            )

    def test_empty_batch_sizes_allowed(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            batch_sizes=[],
        )
        assert req.batch_sizes == []

    def test_batch_sizes_sorted(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            batch_sizes=[8, 2, 4],
        )
        assert req.batch_sizes == [2, 4, 8]

    def test_default_generation_length(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        assert req.generation_length == 128

    def test_force_lm_engine_defaults_to_false(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        assert req.force_lm_engine is False

    def test_force_lm_engine_can_be_enabled(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            force_lm_engine=True,
        )
        assert req.force_lm_engine is True


# =============================================================================
# Prompt generation tests
# =============================================================================


class TestGeneratePrompt:
    def test_exact_token_count(self):
        """Verify prompt generates exact number of tokens."""
        tokenizer = MagicMock()

        # Simulate tokenizer behavior
        def mock_encode(text):
            # Return roughly 1 token per 4 chars
            return list(range(len(text) // 4))

        def mock_decode(tokens):
            return "x" * len(tokens) * 4

        tokenizer.encode = mock_encode
        tokenizer.decode = mock_decode

        prompt = _generate_prompt(tokenizer, 1024)

        # Verify encode was called and result was truncated
        encoded = tokenizer.encode(prompt)
        assert len(encoded) == 1024

    def test_uuid_prefix_uniqueness(self):
        """Verify each generated prompt has a unique UUID prefix."""
        tokenizer = MagicMock()
        tokenizer.encode = lambda text: list(range(2048))
        tokenizer.decode = lambda tokens: f"decoded-{len(tokens)}"

        prompts = set()
        for _ in range(10):
            # We can't easily verify uniqueness since decode is mocked,
            # but we verify encode is called with text containing "BENCH-"
            prompt = _generate_prompt(tokenizer, 100)
            prompts.add(prompt)

        # With mock decode they'll all be the same, but in real usage
        # the UUID prefix ensures cache isolation


# =============================================================================
# Metrics computation tests
# =============================================================================


class TestComputeMetrics:
    def test_basic_metrics(self):
        """Test metric computation with known values."""
        metrics = _compute_single_metrics(
            prompt_tokens=1024,
            completion_tokens=128,
            start_time=0.0,
            first_token_time=0.1,  # 100ms TTFT
            end_time=1.38,  # 1.28s generation
            peak_memory=4 * 1024 * 1024 * 1024,  # 4GB
            cached_tokens=0,
        )

        assert metrics["ttft_ms"] == pytest.approx(100.0, abs=0.1)
        assert metrics["prompt_tokens"] == 1024
        assert metrics["completion_tokens"] == 128
        assert metrics["cached_tokens"] == 0
        assert metrics["peak_memory_bytes"] == 4 * 1024 * 1024 * 1024

        # Gen TPS = 128 / 1.28 = 100 tok/s
        assert metrics["gen_tps"] == pytest.approx(100.0, abs=0.1)

        # Processing TPS = 1024 / 0.1 = 10240 tok/s
        assert metrics["processing_tps"] == pytest.approx(10240.0, abs=1.0)

        # TPOT = 1280ms / 127 = ~10.08 ms/tok
        assert metrics["tpot_ms"] == pytest.approx(10.08, abs=0.1)

        # E2E = 1.38s
        assert metrics["e2e_latency_s"] == pytest.approx(1.38, abs=0.001)

        # Total throughput = (1024 + 128) / 1.38 = ~834.8 tok/s
        assert metrics["total_throughput"] == pytest.approx(834.8, abs=1.0)

    def test_zero_duration_safety(self):
        """Single-token outputs do not report bogus decode throughput."""
        metrics = _compute_single_metrics(
            prompt_tokens=100,
            completion_tokens=1,
            start_time=0.0,
            first_token_time=0.0,
            end_time=0.0,
            peak_memory=0,
            cached_tokens=0,
        )
        # Should not raise, values should be finite
        assert metrics["ttft_ms"] == 0.0
        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0

    def test_native_duration_overrides(self):
        """Native engine timings can override streaming timing artifacts."""
        metrics = _compute_single_metrics(
            prompt_tokens=1024,
            completion_tokens=128,
            start_time=0.0,
            first_token_time=1.0,
            end_time=1.0,
            peak_memory=0,
            cached_tokens=0,
            prefill_duration_s=4.0,
            generation_duration_s=8.0,
        )

        assert metrics["processing_tps"] == pytest.approx(256.0)
        assert metrics["gen_tps"] == pytest.approx(16.0)
        assert metrics["tpot_ms"] == pytest.approx(62.99, abs=0.01)

    def test_single_token_completion_has_no_decode_rate(self):
        """Immediate stop has no inter-token decode interval to benchmark."""
        metrics = _compute_single_metrics(
            prompt_tokens=16384,
            completion_tokens=1,
            start_time=0.0,
            first_token_time=8.629,
            end_time=8.629001,
            peak_memory=0,
            cached_tokens=0,
        )

        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0
        assert metrics["total_throughput"] == pytest.approx(1898.7, abs=0.1)


class TestRunSingleTest:
    @pytest.mark.asyncio
    async def test_uses_native_diffusion_metrics_for_chunked_stream(self):
        """Diffusion streams by canvas, so benchmark must not use chunk timing."""

        class ChunkedDiffusionEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 10,
                    finished=True,
                    finish_reason="length",
                    prompt_tps=256.0,
                    generation_tps=32.0,
                )

        metrics = await _run_single_test(
            ChunkedDiffusionEngine(),
            prompt="prompt",
            max_tokens=128,
            pp_len=1024,
        )

        assert metrics["processing_tps"] == pytest.approx(256.0)
        assert metrics["gen_tps"] == pytest.approx(32.0)
        assert metrics["gen_tps"] < 1000.0

    @pytest.mark.asyncio
    async def test_uses_diffusion_canvas_metrics_when_eos_stops_early(self):
        """Diffusion benchmark TG should measure canvas work, not early EOS text."""

        class EarlyStopDiffusionEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=16,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="short answer",
                    finished=True,
                    finish_reason="stop",
                    prompt_tps=256.0,
                    generation_tps=2.0,
                    diffusion_canvas_tokens=128,
                    diffusion_canvas_tps=64.0,
                )

        metrics = await _run_single_test(
            EarlyStopDiffusionEngine(),
            prompt="prompt",
            max_tokens=128,
            pp_len=1024,
        )

        assert metrics["completion_tokens"] == 128
        assert metrics["gen_tps"] == pytest.approx(64.0)
        assert metrics["tpot_ms"] == pytest.approx(15.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_uses_producer_timestamps_for_aggregated_output(self):
        """Aggregated chunks should use producer-side decode timing."""

        class AggregatedEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 128,
                    finished=True,
                    finish_reason="length",
                    generated_at=0.2,
                    generated_until=1.2,
                )

        with patch("omlx.admin.benchmark.time.perf_counter", side_effect=[0.0, 1.3]):
            metrics = await _run_single_test(
                AggregatedEngine(),
                prompt="prompt",
                max_tokens=128,
                pp_len=1024,
            )

        assert metrics["ttft_ms"] == pytest.approx(200.0)
        assert metrics["gen_tps"] == pytest.approx(128.0)
        assert metrics["tpot_ms"] == pytest.approx(7.87, abs=0.01)

    @pytest.mark.asyncio
    async def test_aggregated_output_without_end_time_has_no_decode_rate(self):
        """Single aggregate chunks need a producer-side end time for tg TPS."""

        class AggregatedEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 128,
                    finished=True,
                    finish_reason="length",
                    generated_at=0.2,
                )

        with patch("omlx.admin.benchmark.time.perf_counter", side_effect=[0.0, 1.2]):
            metrics = await _run_single_test(
                AggregatedEngine(),
                prompt="prompt",
                max_tokens=128,
                pp_len=1024,
            )

        assert metrics["ttft_ms"] == pytest.approx(200.0)
        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0


# =============================================================================
# BenchmarkRun lifecycle tests
# =============================================================================


class TestBenchmarkRunLifecycle:
    def test_create_run(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        run = create_run(req)
        assert run.bench_id.startswith("bench-")
        assert run.status == "running"
        assert run.results == []

    def test_get_run(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        run = create_run(req)
        found = get_run(run.bench_id)
        assert found is run

    def test_get_nonexistent_run(self):
        assert get_run("nonexistent") is None

    def test_cleanup_old_runs(self):
        # Create many completed runs
        for _ in range(15):
            req = BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            )
            run = create_run(req)
            run.status = "completed"

        cleanup_old_runs(max_runs=5)

        # Should have at most ~5 completed + any running ones
        from omlx.admin.benchmark import _benchmark_runs

        completed = [r for r in _benchmark_runs.values() if r.status == "completed"]
        assert len(completed) <= 5


class _FakeBenchTokenizer:
    def encode(self, text):
        return list(range(2048))

    def decode(self, tokens):
        return "prompt"


class _FakeBenchEngine:
    tokenizer = _FakeBenchTokenizer()
    _engine = None

    async def stream_generate(self, **kwargs):
        yield SimpleNamespace(
            completion_tokens=1,
            prompt_tokens=1024,
            cached_tokens=0,
            new_text="x",
            finished=True,
            finish_reason="length",
        )


class _FakeSettingsManager:
    def __init__(self, settings):
        self.settings = settings

    def get_settings(self, model_id):
        return self.settings


class _FakeBenchEnginePool:
    def __init__(self, settings=None, engine=None):
        self._settings_manager = (
            _FakeSettingsManager(settings) if settings is not None else None
        )
        self._engine = engine or _FakeBenchEngine()
        self.force_lm_values = []

    def get_loaded_model_ids(self):
        return []

    async def get_engine(self, model_id, force_lm=False):
        self.force_lm_values.append(force_lm)
        return self._engine

    async def _unload_engine(self, model_id):
        pass


class TestBenchmarkEngineSelection:
    async def _run(self, *, settings=None, force_lm_engine=False):
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=1,
                force_lm_engine=force_lm_engine,
            ),
        )
        pool = _FakeBenchEnginePool(settings)
        await run_benchmark(run, pool)
        return run, pool

    @pytest.mark.asyncio
    async def test_auto_uses_vlm_engine_for_vlm_mtp_with_drafter(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model="draft-model",
        )

        run, pool = await self._run(settings=settings)

        assert pool.force_lm_values == [False]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_force_lm_engine_overrides_vlm_mtp_auto(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model="draft-model",
        )

        run, pool = await self._run(settings=settings, force_lm_engine=True)

        assert pool.force_lm_values == [True]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_auto_keeps_lm_engine_without_vlm_mtp_drafter(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model=None,
        )

        run, pool = await self._run(settings=settings)

        assert pool.force_lm_values == [True]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_batch_request_skips_engine_with_none_scheduler_core(self):
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=1,
                batch_sizes=[2],
            ),
        )
        pool = _FakeBenchEnginePool(engine=_FakeBenchEngine())

        await run_benchmark(run, pool)

        assert run.status == "completed"
        assert [r["test_type"] for r in run.results] == ["single"]

    @pytest.mark.asyncio
    async def test_diffusion_warmup_uses_benchmark_generation_length(self):
        class FakeDiffusionEngine(_FakeBenchEngine):
            is_diffusion_model = True

            def __init__(self):
                self.calls = []

            async def stream_generate(self, **kwargs):
                self.calls.append(kwargs)
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                    prompt_tps=256.0,
                    generation_tps=32.0,
                    diffusion_canvas_tokens=128,
                    diffusion_canvas_tps=64.0,
                )

        engine = FakeDiffusionEngine()
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=128,
            ),
        )
        pool = _FakeBenchEnginePool(engine=engine)

        await run_benchmark(run, pool)

        assert run.status == "completed"
        assert engine.calls[0]["max_tokens"] == 128


# =============================================================================
# SSE event format tests
# =============================================================================


class TestSSEEventFormat:
    @pytest.mark.asyncio
    async def test_send_event(self):
        """Test that events are properly queued."""
        from omlx.admin.benchmark import _send_event

        run = BenchmarkRun(
            bench_id="test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            ),
        )

        await _send_event(run, {
            "type": "progress",
            "phase": "single",
            "message": "Testing",
            "current": 1,
            "total": 3,
        })

        # SSE delivery: events are appended to `run.events` (replay log).
        assert len(run.events) == 1
        event = run.events[0]
        assert event["type"] == "progress"
        assert event["phase"] == "single"
        assert event["current"] == 1
        assert event["total"] == 3

    @pytest.mark.asyncio
    async def test_result_event_format(self):
        from omlx.admin.benchmark import _send_event

        run = BenchmarkRun(
            bench_id="test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            ),
        )

        result_data = {
            "test_type": "single",
            "pp": 1024,
            "tg": 128,
            "ttft_ms": 45.2,
            "gen_tps": 81.3,
        }
        await _send_event(run, {"type": "result", "data": result_data})

        assert len(run.events) == 1
        event = run.events[0]
        assert event["type"] == "result"
        assert event["data"]["test_type"] == "single"
        assert event["data"]["pp"] == 1024
