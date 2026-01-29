"""Tests for the soil_moisture module."""

import numpy as np
import pytest

from dolphin.soil_moisture import (
    SoilMoistureOutput,
    compute_cumulative_closure_phase,
    compute_soil_moisture_index,
    detrend_cumulative_closure_phase,
)


class TestComputeCumulativeClosurePhase:
    """Tests for compute_cumulative_closure_phase function."""

    def test_basic_cumsum(self):
        """Test basic cumulative sum computation."""
        closure_phases = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_cumulative_closure_phase(closure_phases)
        expected = np.array([0.1, 0.3, 0.6, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_3d_array(self):
        """Test with 3D array (n_triplets, rows, cols)."""
        shape = (5, 10, 10)
        closure_phases = np.random.randn(*shape) * 0.1
        result = compute_cumulative_closure_phase(closure_phases, axis=0)

        assert result.shape == shape
        # Check that the last slice is the sum
        np.testing.assert_allclose(result[-1], closure_phases.sum(axis=0), atol=1e-10)

    def test_axis_parameter(self):
        """Test cumsum along different axes."""
        shape = (5, 3)
        arr = np.random.randn(*shape)

        result_axis0 = compute_cumulative_closure_phase(arr, axis=0)
        result_axis1 = compute_cumulative_closure_phase(arr, axis=1)

        np.testing.assert_allclose(result_axis0, np.cumsum(arr, axis=0), atol=1e-10)
        np.testing.assert_allclose(result_axis1, np.cumsum(arr, axis=1), atol=1e-10)


class TestDetrendCumulativeClosurePhase:
    """Tests for detrend_cumulative_closure_phase function."""

    def test_removes_linear_trend(self):
        """Test that a pure linear trend is removed."""
        n_times = 20
        t = np.arange(n_times, dtype=float)
        # Pure linear signal: y = 2*t + 3
        linear_signal = 2.0 * t + 3.0

        detrended = detrend_cumulative_closure_phase(linear_signal)

        # After removing trend, should be close to zero (or constant)
        # The mean might not be exactly zero but the variation should be minimal
        np.testing.assert_allclose(np.std(detrended), 0.0, atol=1e-10)

    def test_preserves_non_linear_signal(self):
        """Test that non-linear components are preserved."""
        n_times = 50
        t = np.arange(n_times, dtype=float)
        # Signal with linear trend + higher-frequency sinusoidal component
        # Use multiple cycles so the sinusoidal doesn't bias the slope estimate
        linear_part = 0.5 * t
        sinusoidal_part = np.sin(2 * np.pi * t / 10)  # ~5 cycles
        signal = linear_part + sinusoidal_part

        detrended = detrend_cumulative_closure_phase(signal)

        # The sinusoidal part should be mostly preserved
        # Correlation between detrended and sinusoidal should be high
        correlation = np.corrcoef(detrended, sinusoidal_part)[0, 1]
        assert abs(correlation) > 0.95

    def test_return_trend(self):
        """Test that trend is returned when requested."""
        n_times = 20
        t = np.arange(n_times, dtype=float)
        slope = 2.5
        signal = slope * t + 10.0  # Linear signal

        detrended, trend = detrend_cumulative_closure_phase(
            signal, return_trend=True
        )

        # Trend should be close to the actual slope
        np.testing.assert_allclose(trend, slope, atol=1e-10)

    def test_3d_array(self):
        """Test detrending 3D array."""
        shape = (10, 5, 5)
        n_times = shape[0]
        t = np.arange(n_times, dtype=float)

        # Create array with different slopes per pixel
        slopes = np.random.randn(5, 5) * 0.1
        signal = t[:, np.newaxis, np.newaxis] * slopes[np.newaxis, :, :]

        detrended, trend = detrend_cumulative_closure_phase(
            signal, axis=0, return_trend=True
        )

        assert detrended.shape == shape
        assert trend.shape == (5, 5)
        # Trend should match the slopes we used
        np.testing.assert_allclose(trend, slopes, atol=1e-10)

    def test_handles_nan_values(self):
        """Test that NaN values are handled gracefully."""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # Should not raise an error
        detrended = detrend_cumulative_closure_phase(signal)
        assert detrended.shape == signal.shape


class TestComputeSoilMoistureIndex:
    """Tests for compute_soil_moisture_index function."""

    def test_basic_computation(self):
        """Test basic soil moisture index computation."""
        shape = (10, 20, 20)
        closure_phases = np.random.randn(*shape) * 0.1

        result = compute_soil_moisture_index(closure_phases)

        assert isinstance(result, SoilMoistureOutput)
        assert result.ismi.shape == shape
        assert result.cumulative_closure_phase.shape == shape
        assert result.trend.shape == (20, 20)

    def test_with_temporal_coherence_mask(self):
        """Test masking based on temporal coherence."""
        shape = (10, 20, 20)
        closure_phases = np.random.randn(*shape) * 0.1

        # Create temporal coherence with some low values
        temporal_coherence = np.ones((20, 20))
        temporal_coherence[5:10, 5:10] = 0.3  # Low coherence region

        result = compute_soil_moisture_index(
            closure_phases,
            temporal_coherence=temporal_coherence,
            coherence_threshold=0.5,
        )

        # Low coherence pixels should be NaN
        assert np.all(np.isnan(result.ismi[:, 5:10, 5:10]))
        # High coherence pixels should be finite
        assert np.all(np.isfinite(result.ismi[:, 0:5, 0:5]))

    def test_cumulative_is_cumsum(self):
        """Verify cumulative closure phase is actually cumulative sum."""
        shape = (8, 10, 10)
        closure_phases = np.random.randn(*shape) * 0.1

        result = compute_soil_moisture_index(closure_phases)

        expected_cumulative = np.cumsum(closure_phases, axis=0)
        np.testing.assert_allclose(
            result.cumulative_closure_phase, expected_cumulative, atol=1e-10
        )

    def test_single_pixel(self):
        """Test with single pixel (1D array)."""
        closure_phases = np.array([0.1, -0.05, 0.08, -0.03, 0.12])

        result = compute_soil_moisture_index(closure_phases)

        assert result.ismi.shape == closure_phases.shape
        assert np.isscalar(result.trend) or result.trend.shape == ()


class TestSoilMoistureOutputStructure:
    """Tests for SoilMoistureOutput named tuple."""

    def test_named_tuple_fields(self):
        """Test that SoilMoistureOutput has expected fields."""
        output = SoilMoistureOutput(
            ismi=np.array([1, 2, 3]),
            cumulative_closure_phase=np.array([1, 2, 3]),
            trend=np.array(0.5),
        )

        assert hasattr(output, "ismi")
        assert hasattr(output, "cumulative_closure_phase")
        assert hasattr(output, "trend")

    def test_unpacking(self):
        """Test that output can be unpacked."""
        output = SoilMoistureOutput(
            ismi=np.array([1]),
            cumulative_closure_phase=np.array([2]),
            trend=np.array(3),
        )

        ismi, cumulative, trend = output
        np.testing.assert_array_equal(ismi, [1])
        np.testing.assert_array_equal(cumulative, [2])
        np.testing.assert_array_equal(trend, 3)


class TestPhysicalBehavior:
    """Tests verifying expected physical behavior of soil moisture index."""

    def test_positive_trend_removal(self):
        """Test that positive systematic trends are removed."""
        n_times = 20
        t = np.arange(n_times)

        # Simulate closure phases with positive bias (e.g., systematic drift)
        closure_phases = 0.05 * np.ones(n_times)  # Constant positive bias

        result = compute_soil_moisture_index(closure_phases)

        # The detrended result should have near-zero mean
        assert abs(np.nanmean(result.ismi)) < 0.1

    def test_step_change_creates_variation(self):
        """Test that step changes (soil moisture events) create signal variation.

        A step change in closure phase becomes a ramp in cumulative closure phase.
        After detrending, this manifests as variance in the ISMI signal.
        """
        n_times = 30
        closure_phases = np.zeros(n_times)

        # Simulate a rain event: sudden increase in closure phase
        closure_phases[15:] = 0.3  # Step change at time 15

        result = compute_soil_moisture_index(closure_phases)

        # A step change should create significant variance in the ISMI
        # (unlike a constant signal which would have zero variance after detrending)
        assert np.var(result.ismi) > 0.1

    def test_constant_closure_phases_zero_variance(self):
        """Test that constant closure phases result in near-zero variance after detrending."""
        n_times = 30
        # Constant closure phase (no soil moisture change)
        closure_phases = 0.1 * np.ones(n_times)

        result = compute_soil_moisture_index(closure_phases)

        # After detrending a linear cumulative (from constant closure phases),
        # the variance should be effectively zero
        assert np.var(result.ismi) < 1e-10
