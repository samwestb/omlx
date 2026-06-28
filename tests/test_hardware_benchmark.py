# SPDX-License-Identifier: Apache-2.0
"""Tests for hardware detection functions used in omlx.ai benchmark integration."""

from unittest.mock import MagicMock, patch

import pytest

from omlx.utils.hardware import (
    get_chip_name,
    get_gpu_core_count,
    get_os_version,
    get_total_memory_bytes,
    parse_chip_info,
)


class TestParseChipInfo:
    def test_m4_pro(self):
        assert parse_chip_info("Apple M4 Pro") == ("M4", "Pro")

    def test_m3_max(self):
        assert parse_chip_info("Apple M3 Max") == ("M3", "Max")

    def test_m2_ultra(self):
        assert parse_chip_info("Apple M2 Ultra") == ("M2", "Ultra")

    def test_m1_base(self):
        assert parse_chip_info("Apple M1") == ("M1", "")

    def test_m4_base(self):
        assert parse_chip_info("Apple M4") == ("M4", "")

    def test_m5_pro(self):
        assert parse_chip_info("Apple M5 Pro") == ("M5", "Pro")

    def test_fallback(self):
        assert parse_chip_info("Apple Silicon") == ("M1", "")

    def test_empty_string(self):
        assert parse_chip_info("") == ("M1", "")


class TestSystemToolsUseAbsolutePath:
    """System tools must be invoked by absolute path.

    They live in /usr/sbin, which is not on PATH in some headless launchd
    contexts (e.g. `brew services`). A bare name would raise FileNotFoundError
    there and silently degrade detection (chip -> M1). See issue #1322.
    """

    def _cmd_of(self, mock_run):
        return mock_run.call_args[0][0]

    def test_get_chip_name_absolute_path(self):
        with patch("omlx.utils.hardware.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Apple M4 Pro\n")
            assert get_chip_name() == "Apple M4 Pro"
            assert self._cmd_of(mock_run)[0] == "/usr/sbin/sysctl"

    def test_get_total_memory_absolute_path(self):
        with patch("omlx.utils.hardware.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="68719476736\n")
            assert get_total_memory_bytes() == 68719476736
            assert self._cmd_of(mock_run)[0] == "/usr/sbin/sysctl"

    def test_get_gpu_core_count_absolute_path(self):
        with patch("omlx.utils.hardware.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="      Total Number of Cores: 40\n"
            )
            assert get_gpu_core_count() == 40
            assert self._cmd_of(mock_run)[0] == "/usr/sbin/system_profiler"

    def test_chip_name_falls_back_when_tool_missing(self):
        # Simulates /usr/sbin not on PATH (FileNotFoundError) -> M1 fallback,
        # which is exactly the #1322 symptom the absolute path prevents.
        with patch("omlx.utils.hardware.subprocess.run", side_effect=FileNotFoundError):
            assert get_chip_name() == "Apple Silicon"
            assert parse_chip_info(get_chip_name()) == ("M1", "")


class TestGetOsVersion:
    def test_returns_string(self):
        result = get_os_version()
        assert isinstance(result, str)
        assert result.startswith("macOS")
