"""
Hyperparameter Selector for AutoDD

Implements a cascading (hierarchical) selection strategy:
1. Level 1: LinUCB selects the detector based on meta-features
2. Level 2: Thompson Sampling selects hyperparameters for the chosen detector

This module provides adaptive hyperparameter selection that learns which
configurations work best for each detector over time.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .detector_portfolio import DetectorPortfolio, DETECTOR_PARAM_RANGES


class ThompsonSamplingConfig:
    """
    Thompson Sampling for hyperparameter configuration selection.
    
    Maintains Beta distributions for each configuration to balance
    exploration and exploitation in hyperparameter space.
    
    Parameters
    ----------
    n_configs : int
        Number of configurations to maintain per detector.
    prior_alpha : float
        Prior alpha for Beta distribution (successes).
    prior_beta : float
        Prior beta for Beta distribution (failures).
    """
    
    def __init__(
        self,
        n_configs: int = 5,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.n_configs = n_configs
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Beta distribution parameters for each config
        self.alphas = np.full(n_configs, prior_alpha)
        self.betas = np.full(n_configs, prior_beta)
        
        # Track selection counts
        self.selection_counts = np.zeros(n_configs)
        self.total_rewards = np.zeros(n_configs)
    
    def select(self, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select a configuration using Thompson Sampling.
        
        Returns
        -------
        config_idx : int
            Index of selected configuration.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample from Beta distributions
        samples = rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))
    
    def update(self, config_idx: int, reward: float):
        """
        Update the Beta distribution for a configuration.
        
        Parameters
        ----------
        config_idx : int
            Index of the configuration that was used.
        reward : float
            Reward received (will be normalized to [0, 1]).
        """
        # Normalize reward to [0, 1] for Beta distribution
        normalized_reward = (reward + 1.0) / 2.0  # Assuming reward in [-1, 1]
        normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
        
        # Update counts
        self.selection_counts[config_idx] += 1
        self.total_rewards[config_idx] += reward
        
        # Update Beta distribution (Bernoulli-like update)
        self.alphas[config_idx] += normalized_reward
        self.betas[config_idx] += (1.0 - normalized_reward)
    
    def get_best_config(self) -> int:
        """Get the configuration with highest mean reward."""
        means = self.alphas / (self.alphas + self.betas)
        return int(np.argmax(means))
    
    def get_stats(self) -> Dict:
        """Get statistics for all configurations."""
        means = self.alphas / (self.alphas + self.betas)
        return {
            'means': means.tolist(),
            'selection_counts': self.selection_counts.tolist(),
            'total_rewards': self.total_rewards.tolist()
        }
    
    def reset(self):
        """Reset to prior distributions."""
        self.alphas = np.full(self.n_configs, self.prior_alpha)
        self.betas = np.full(self.n_configs, self.prior_beta)
        self.selection_counts = np.zeros(self.n_configs)
        self.total_rewards = np.zeros(self.n_configs)


class CascadingHyperparameterSelector:
    """
    Cascading (Hierarchical) Hyperparameter Selector for AutoDD.
    
    Implements a two-level selection strategy:
    - Level 1: Detector selection (handled by LinUCB in main AutoDD)
    - Level 2: Hyperparameter selection for chosen detector (Thompson Sampling)
    
    For each detector, maintains a pool of hyperparameter configurations and
    uses Thompson Sampling to adaptively select the best configuration.
    
    Parameters
    ----------
    portfolio : DetectorPortfolio
        The detector portfolio.
    n_configs_per_detector : int, default=5
        Number of hyperparameter configurations per detector.
    include_default : bool, default=True
        Whether to include default configuration (index 0) for each detector.
    seed : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    configs : dict
        Maps detector names to lists of hyperparameter configurations.
    samplers : dict
        Maps detector names to ThompsonSamplingConfig instances.
    
    Examples
    --------
    >>> portfolio = DetectorPortfolio()
    >>> hp_selector = CascadingHyperparameterSelector(portfolio, n_configs_per_detector=5)
    >>> config_idx, params = hp_selector.select_config('ADWIN')
    >>> # ... run detection with params ...
    >>> hp_selector.update('ADWIN', config_idx, reward=0.8)
    """
    
    def __init__(
        self,
        portfolio: DetectorPortfolio,
        n_configs_per_detector: int = 5,
        include_default: bool = True,
        seed: Optional[int] = None
    ):
        self.portfolio = portfolio
        self.n_configs = n_configs_per_detector
        self.include_default = include_default
        self.rng = np.random.default_rng(seed)
        
        # Generate configurations for each detector
        self.configs: Dict[str, List[Dict[str, Any]]] = {}
        self.samplers: Dict[str, ThompsonSamplingConfig] = {}
        
        self._initialize_configs()
        
        # Track current selections
        self.current_config_idx: Dict[str, int] = {}
        self.current_params: Dict[str, Dict[str, Any]] = {}
        
        # History
        self.selection_history: List[Tuple[str, int]] = []
    
    def _initialize_configs(self):
        """Initialize hyperparameter configurations for each detector."""
        for detector_name in self.portfolio.detector_names:
            # Generate configurations
            configs = self.portfolio.get_detector_configurations(
                detector_name,
                n_configs=self.n_configs,
                include_default=self.include_default,
                rng=self.rng
            )
            self.configs[detector_name] = configs
            
            # Initialize Thompson Sampling for this detector
            self.samplers[detector_name] = ThompsonSamplingConfig(
                n_configs=len(configs)
            )
    
    def select_config(self, detector_name: str) -> Tuple[int, Dict[str, Any]]:
        """
        Select a hyperparameter configuration for the given detector.
        
        Uses Thompson Sampling to balance exploration and exploitation.
        
        Parameters
        ----------
        detector_name : str
            Name of the detector.
        
        Returns
        -------
        config_idx : int
            Index of selected configuration.
        params : dict
            The hyperparameter configuration.
        """
        if detector_name not in self.samplers:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Thompson Sampling selection
        sampler = self.samplers[detector_name]
        config_idx = sampler.select(self.rng)
        params = self.configs[detector_name][config_idx]
        
        # Track current selection
        self.current_config_idx[detector_name] = config_idx
        self.current_params[detector_name] = params
        self.selection_history.append((detector_name, config_idx))
        
        return config_idx, params
    
    def update(self, detector_name: str, config_idx: int, reward: float):
        """
        Update the Thompson Sampling model for a detector configuration.
        
        Parameters
        ----------
        detector_name : str
            Name of the detector.
        config_idx : int
            Index of the configuration that was used.
        reward : float
            Reward received for this configuration.
        """
        if detector_name not in self.samplers:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        self.samplers[detector_name].update(config_idx, reward)
    
    def get_best_config(self, detector_name: str) -> Tuple[int, Dict[str, Any]]:
        """
        Get the best configuration for a detector based on learned rewards.
        
        Parameters
        ----------
        detector_name : str
            Name of the detector.
        
        Returns
        -------
        config_idx : int
            Index of best configuration.
        params : dict
            The best hyperparameter configuration.
        """
        best_idx = self.samplers[detector_name].get_best_config()
        return best_idx, self.configs[detector_name][best_idx]
    
    def create_detector(self, detector_name: str, params: Dict[str, Any]) -> Any:
        """
        Create a detector instance with specified hyperparameters.
        
        Parameters
        ----------
        detector_name : str
            Name of the detector.
        params : dict
            Hyperparameter configuration.
        
        Returns
        -------
        detector : drift detector
            New detector instance.
        """
        return self.portfolio.create_detector_with_params(detector_name, params)
    
    def get_config_stats(self, detector_name: str) -> Dict:
        """Get statistics for a detector's configurations."""
        stats = self.samplers[detector_name].get_stats()
        stats['configs'] = self.configs[detector_name]
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all detectors."""
        return {name: self.get_config_stats(name) 
                for name in self.portfolio.detector_names}
    
    def get_selection_distribution(self) -> Dict[str, Dict[int, float]]:
        """
        Get the distribution of configuration selections per detector.
        
        Returns
        -------
        distribution : dict
            Maps detector names to config selection frequencies.
        """
        dist = defaultdict(lambda: defaultdict(int))
        
        for detector_name, config_idx in self.selection_history:
            dist[detector_name][config_idx] += 1
        
        # Normalize
        result = {}
        for detector_name, counts in dist.items():
            total = sum(counts.values())
            result[detector_name] = {
                idx: count / total for idx, count in counts.items()
            }
        
        return result
    
    def reset(self):
        """Reset all Thompson Sampling models."""
        for sampler in self.samplers.values():
            sampler.reset()
        self.current_config_idx = {}
        self.current_params = {}
        self.selection_history = []
    
    def save_state(self) -> Dict:
        """Save selector state."""
        return {
            'configs': self.configs,
            'sampler_states': {
                name: {
                    'alphas': sampler.alphas.tolist(),
                    'betas': sampler.betas.tolist(),
                    'selection_counts': sampler.selection_counts.tolist(),
                    'total_rewards': sampler.total_rewards.tolist()
                }
                for name, sampler in self.samplers.items()
            },
            'selection_history': self.selection_history
        }
    
    def load_state(self, state: Dict):
        """Load selector state."""
        self.configs = state['configs']
        for name, sampler_state in state['sampler_states'].items():
            self.samplers[name].alphas = np.array(sampler_state['alphas'])
            self.samplers[name].betas = np.array(sampler_state['betas'])
            self.samplers[name].selection_counts = np.array(sampler_state['selection_counts'])
            self.samplers[name].total_rewards = np.array(sampler_state['total_rewards'])
        self.selection_history = state['selection_history']
