"""
Tests for uncertainty quantification, benchmark suite, and ensemble modules.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, output_shape=(3, 64, 64), add_noise=True):
        super().__init__()
        self.output_shape = output_shape
        self.add_noise = add_noise
        self.linear = nn.Linear(10, 10)  # Dummy param
        
    def forward(self, x):
        return torch.randn(x.shape[0], *self.output_shape)
    
    def draw_samples(
        self,
        conditioning,
        batch_size=1,
        n_sampling_steps=None,
        param_conditioning=None,
    ):
        output = torch.randn(batch_size, *self.output_shape)
        if self.add_noise:
            # Add small noise so samples differ
            output = output + 0.1 * torch.randn_like(output)
        return output


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator."""
    
    @pytest.fixture
    def model(self):
        return MockModel()
    
    @pytest.fixture
    def conditioning(self):
        return torch.randn(2, 4, 64, 64)
    
    def test_predict_with_uncertainty_basic(self, model, conditioning):
        """Test basic uncertainty prediction."""
        from vdm.uncertainty import UncertaintyEstimator
        
        estimator = UncertaintyEstimator(model, n_samples=5)
        result = estimator.predict_with_uncertainty(
            conditioning, 
            show_progress=False,
            return_samples=True
        )
        
        # Check shapes
        assert result.mean.shape == (2, 3, 64, 64)
        assert result.std.shape == (2, 3, 64, 64)
        assert result.samples.shape == (5, 2, 3, 64, 64)
        
        # Std should be > 0 (samples should differ)
        assert result.std.mean() > 0
    
    def test_predict_with_params(self, model, conditioning):
        """Test prediction with parameter conditioning."""
        from vdm.uncertainty import UncertaintyEstimator
        
        estimator = UncertaintyEstimator(model, n_samples=3)
        params = torch.randn(2, 6)
        
        result = estimator.predict_with_uncertainty(
            conditioning,
            params=params,
            show_progress=False
        )
        
        assert result.mean.shape == (2, 3, 64, 64)
    
    def test_coefficient_of_variation(self, model, conditioning):
        """Test CV computation."""
        from vdm.uncertainty import UncertaintyEstimator
        
        estimator = UncertaintyEstimator(model, n_samples=5)
        result = estimator.predict_with_uncertainty(
            conditioning,
            show_progress=False
        )
        
        # CV should be std / |mean|
        assert result.coefficient_of_variation is not None
        assert result.coefficient_of_variation.shape == result.mean.shape
    
    def test_mc_dropout_flag(self, conditioning):
        """Test MC Dropout enabling."""
        from vdm.uncertainty import UncertaintyEstimator
        
        # Model with dropout
        class ModelWithDropout(MockModel):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.1)
        
        model = ModelWithDropout()
        estimator = UncertaintyEstimator(model, n_samples=3, use_mc_dropout=True)
        
        result = estimator.predict_with_uncertainty(
            conditioning,
            show_progress=False
        )
        
        assert result.mean is not None


class TestCalibration:
    """Tests for calibration metrics."""
    
    def test_compute_calibration(self):
        """Test calibration computation."""
        from vdm.uncertainty import UncertaintyEstimator
        
        model = MockModel()
        estimator = UncertaintyEstimator(model, n_samples=3)
        
        # Create fake predictions
        predictions_mean = np.random.randn(10, 10)
        predictions_std = np.abs(np.random.randn(10, 10)) + 0.1
        ground_truth = predictions_mean + 0.5 * predictions_std * np.random.randn(10, 10)
        
        coverage, ece = estimator.compute_calibration(
            predictions_mean,
            predictions_std,
            ground_truth,
            confidence_levels=[0.5, 0.9]
        )
        
        assert 0.5 in coverage
        assert 0.9 in coverage
        assert 0 <= ece <= 1
    
    def test_reliability_diagram_data(self):
        """Test reliability diagram data generation."""
        from vdm.uncertainty import UncertaintyEstimator
        
        model = MockModel()
        estimator = UncertaintyEstimator(model, n_samples=3)
        
        predictions_mean = np.random.randn(20, 20)
        predictions_std = np.abs(np.random.randn(20, 20)) + 0.1
        ground_truth = predictions_mean + predictions_std * np.random.randn(20, 20)
        
        expected, observed, counts = estimator.reliability_diagram_data(
            predictions_mean,
            predictions_std,
            ground_truth,
            n_bins=10
        )
        
        assert len(expected) == 10
        assert len(observed) == 10
        assert len(counts) == 10


class TestEnsembleUncertainty:
    """Tests for EnsembleUncertainty class."""
    
    def test_ensemble_basic(self):
        """Test basic ensemble prediction."""
        from vdm.uncertainty import EnsembleUncertainty
        
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleUncertainty(models)
        
        conditioning = torch.randn(2, 4, 64, 64)
        result = ensemble.predict_with_uncertainty(
            conditioning,
            show_progress=False
        )
        
        assert result.mean.shape == (2, 3, 64, 64)
        assert result.std.shape == (2, 3, 64, 64)
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        from vdm.uncertainty import EnsembleUncertainty
        
        models = [MockModel() for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleUncertainty(models, weights=weights)
        
        # Weights should be normalized
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6
        
        conditioning = torch.randn(1, 4, 64, 64)
        result = ensemble.predict_with_uncertainty(
            conditioning,
            show_progress=False
        )
        
        assert result.mean is not None
    
    def test_ensemble_disagreement(self):
        """Test pairwise disagreement computation."""
        from vdm.uncertainty import EnsembleUncertainty
        
        models = [MockModel() for _ in range(3)]
        ensemble = EnsembleUncertainty(models)
        
        conditioning = torch.randn(1, 4, 64, 64)
        disagreement = ensemble.compute_ensemble_disagreement(conditioning)
        
        assert disagreement.shape == (3, 3)
        # Diagonal should be 0 (self-comparison)
        np.testing.assert_array_almost_equal(np.diag(disagreement), np.zeros(3), decimal=5)


class TestUncertaintyMaps:
    """Tests for uncertainty map computation."""
    
    def test_std_method(self):
        """Test std uncertainty map."""
        from vdm.uncertainty import compute_uncertainty_maps
        
        samples = np.random.randn(10, 3, 32, 32)
        uncertainty = compute_uncertainty_maps(samples, method='std')
        
        assert uncertainty.shape == (3, 32, 32)
    
    def test_iqr_method(self):
        """Test IQR uncertainty map."""
        from vdm.uncertainty import compute_uncertainty_maps
        
        samples = np.random.randn(10, 3, 32, 32)
        uncertainty = compute_uncertainty_maps(samples, method='iqr')
        
        assert uncertainty.shape == (3, 32, 32)
        assert np.all(uncertainty >= 0)  # IQR is non-negative


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""
    
    @pytest.fixture
    def model(self):
        return MockModel()
    
    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple mock dataloader."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                conditioning = torch.randn(4, 64, 64)
                target = torch.randn(3, 64, 64)
                params = torch.randn(6)
                return conditioning, target, params
        
        return torch.utils.data.DataLoader(SimpleDataset(), batch_size=2)
    
    def test_benchmark_creation(self, model):
        """Test benchmark suite creation."""
        from vdm.benchmark import BenchmarkSuite
        
        suite = BenchmarkSuite(
            models={'test': model},
            device='cpu'
        )
        
        assert 'test' in suite.models
    
    def test_pixel_metrics(self):
        """Test pixel-level metric computation."""
        from vdm.benchmark import BenchmarkSuite
        
        suite = BenchmarkSuite(device='cpu')
        
        prediction = np.random.randn(64, 64)
        target = prediction + 0.1 * np.random.randn(64, 64)
        
        metrics = suite.compute_pixel_metrics(prediction, target)
        
        assert metrics.mse > 0
        assert metrics.rmse > 0
        assert -1 <= metrics.correlation <= 1
    
    def test_mass_metrics(self):
        """Test mass metric computation."""
        from vdm.benchmark import BenchmarkSuite
        
        suite = BenchmarkSuite(device='cpu')
        
        target = np.abs(np.random.randn(64, 64)) + 1
        prediction = target * 1.1  # 10% bias
        
        metrics = suite.compute_mass_metrics(prediction, target)
        
        assert abs(metrics['mass_bias'] - 0.1) < 0.01
    
    def test_quick_benchmark(self):
        """Test quick benchmark function."""
        from vdm.benchmark import quick_benchmark
        
        model = MockModel()
        conditioning = torch.randn(1, 4, 64, 64)
        target = torch.randn(1, 3, 64, 64)
        
        results = quick_benchmark(
            model, 
            conditioning, 
            target[0],
            n_samples=2,
            device='cpu'
        )
        
        assert 'mse' in results
        assert 'correlation' in results
        assert 'time_mean_ms' in results


class TestModelEnsemble:
    """Tests for ModelEnsemble."""
    
    def test_simple_ensemble(self):
        """Test simple averaging ensemble."""
        from vdm.ensemble import ModelEnsemble
        
        models = [MockModel(add_noise=False) for _ in range(3)]
        ensemble = ModelEnsemble(models)
        
        conditioning = torch.randn(2, 4, 64, 64)
        output = ensemble.draw_samples(conditioning, batch_size=2)
        
        assert output.shape == (2, 3, 64, 64)
    
    def test_ensemble_with_individual(self):
        """Test ensemble returning individual predictions."""
        from vdm.ensemble import ModelEnsemble
        
        models = [MockModel() for _ in range(3)]
        ensemble = ModelEnsemble(models)
        
        conditioning = torch.randn(1, 4, 64, 64)
        result = ensemble.draw_samples(
            conditioning, 
            batch_size=1,
            return_individual=True
        )
        
        assert result.mean.shape == (1, 3, 64, 64)
        assert result.std.shape == (1, 3, 64, 64)
        assert len(result.individual) == 3


class TestWeightedEnsemble:
    """Tests for WeightedEnsemble."""
    
    def test_weighted_ensemble_basic(self):
        """Test weighted ensemble."""
        from vdm.ensemble import WeightedEnsemble
        
        models = [MockModel() for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        
        ensemble = WeightedEnsemble(models, weights=weights)
        
        # Check weights are normalized
        assert abs(ensemble.normalized_weights.sum().item() - 1.0) < 1e-6
        
        conditioning = torch.randn(1, 4, 64, 64)
        output = ensemble.draw_samples(conditioning, batch_size=1)
        
        assert output.shape == (1, 3, 64, 64)
    
    def test_learnable_weights(self):
        """Test learnable weights."""
        from vdm.ensemble import WeightedEnsemble
        
        models = [MockModel() for _ in range(3)]
        ensemble = WeightedEnsemble(models, learnable_weights=True)
        
        # Weights should be parameters
        assert hasattr(ensemble, 'weight_logits')
        assert ensemble.weight_logits.requires_grad


class TestChannelWiseEnsemble:
    """Tests for ChannelWiseEnsemble."""
    
    def test_channel_wise_ensemble(self):
        """Test per-channel weighting."""
        from vdm.ensemble import ChannelWiseEnsemble
        
        models = [MockModel() for _ in range(2)]
        
        # Model 0 weighted for channel 0, Model 1 for channels 1,2
        channel_weights = torch.tensor([
            [0.8, 0.2, 0.2],  # Model 0
            [0.2, 0.8, 0.8],  # Model 1
        ])
        
        ensemble = ChannelWiseEnsemble(models, channel_weights=channel_weights)
        
        conditioning = torch.randn(1, 4, 64, 64)
        output = ensemble.draw_samples(conditioning, batch_size=1)
        
        assert output.shape == (1, 3, 64, 64)


class TestDiversityEnsemble:
    """Tests for DiversityEnsemble."""
    
    def test_diversity_loss(self):
        """Test diversity loss computation."""
        from vdm.ensemble import DiversityEnsemble
        
        models = [MockModel() for _ in range(3)]
        ensemble = DiversityEnsemble(models)
        
        # Create predictions
        predictions = [torch.randn(2, 3, 64, 64) for _ in range(3)]
        
        diversity = ensemble.compute_diversity_loss(predictions)
        
        assert diversity.item() >= 0
    
    def test_diversity_ensemble_sampling(self):
        """Test sampling with diversity metric."""
        from vdm.ensemble import DiversityEnsemble
        
        models = [MockModel() for _ in range(3)]
        ensemble = DiversityEnsemble(models)
        
        conditioning = torch.randn(1, 4, 64, 64)
        output, diversity = ensemble.draw_samples(
            conditioning, 
            batch_size=1,
            return_diversity=True
        )
        
        assert output.shape == (1, 3, 64, 64)
        assert isinstance(diversity, float)


class TestIntegration:
    """Integration tests combining modules."""
    
    def test_uncertainty_with_ensemble(self):
        """Test uncertainty estimation with ensemble."""
        from vdm.uncertainty import EnsembleUncertainty
        from vdm.ensemble import ModelEnsemble
        
        models = [MockModel() for _ in range(3)]
        
        # Both should give similar results
        ensemble_unc = EnsembleUncertainty(models)
        ensemble_mod = ModelEnsemble(models)
        
        conditioning = torch.randn(1, 4, 64, 64)
        
        result_unc = ensemble_unc.predict_with_uncertainty(
            conditioning, 
            show_progress=False
        )
        result_mod = ensemble_mod.draw_samples(
            conditioning, 
            batch_size=1,
            return_individual=True
        )
        
        # Both should have mean and std
        assert result_unc.mean.shape == result_mod.mean.shape
        assert result_unc.std.shape == result_mod.std.shape
