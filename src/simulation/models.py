"""
Stochastic Models for Monte Carlo Simulation
===========================================

This module implements various stochastic models for price simulation:
- Geometric Brownian Motion (GBM)
- Heston Stochastic Volatility Model
- SABR-CGMY Model (Stochastic Alpha Beta Rho with CGMY jumps)
"""

import numpy as np
from typing import Optional, Tuple, List
import logging
from scipy.stats import norm, invgauss, levy_stable
from scipy.special import gamma
from scipy import integrate
import concurrent.futures
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

# Determine device once at module load time
try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        TORCH_DTYPE = torch.float32  # Use float32 for both CUDA and MPS
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        TORCH_DTYPE = torch.float32  # MPS only supports float32
        logger.info("Using MPS device")
    else:
        DEVICE = torch.device("cpu")
        TORCH_DTYPE = torch.float64  # CPU can use float64
        logger.info("Using CPU device")
except Exception as e:
    logger.warning(f"Error initializing device, falling back to CPU: {e}")
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float64
    
# Cache for expensive CGMY calculations
CGMY_CACHE = {}

# ============== Geometric Brownian Motion (GBM) ==============

def simulate_gbm(current_price: float, annualized_drift: float, annualized_volatility: float,
                time_horizon: int, num_paths: int) -> np.ndarray:
    """
    Vectorized implementation of Geometric Brownian Motion simulation using GPU if available.
    
    Args:
        current_price: Current stock price
        annualized_drift: Annualized drift (mu)
        annualized_volatility: Annualized volatility (sigma)
        time_horizon: Number of days to simulate
        num_paths: Total number of paths to simulate
        
    Returns:
        Array of final prices for each path
    """
    daily_drift = annualized_drift / 252
    daily_volatility = annualized_volatility / np.sqrt(252)
    
    # Use GPU-accelerated tensors if possible
    if DEVICE.type != 'cpu' and num_paths * time_horizon > 5000:
        try:
            # Move calculations to GPU for large simulations
            # Explicitly set the data type to float32 for all tensors
            noise = torch.randn(num_paths, time_horizon, device=DEVICE, dtype=TORCH_DTYPE) * daily_volatility
            daily_returns = torch.full((num_paths, time_horizon), daily_drift, 
                                     device=DEVICE, dtype=TORCH_DTYPE) + noise
            cumulative_returns = torch.cumsum(daily_returns, dim=1)
            current_price_tensor = torch.tensor(current_price, device=DEVICE, dtype=TORCH_DTYPE)
            price_paths = current_price_tensor * torch.exp(cumulative_returns)
            
            # Return only final prices
            return price_paths[:, -1].cpu().numpy()
        except Exception as e:
            # Fallback to CPU if GPU fails
            logger.warning(f"GPU simulation failed, falling back to CPU: {e}")
    
    # CPU implementation for smaller simulations or as fallback
    noise = np.random.normal(0, 1, size=(num_paths, time_horizon)) * daily_volatility
    daily_returns = np.full((num_paths, time_horizon), daily_drift) + noise
    cumulative_returns = np.cumsum(daily_returns, axis=1)
    price_paths = current_price * np.exp(cumulative_returns)
    
    return price_paths[:, -1]

# ============== Heston Stochastic Volatility Model ==============

def simulate_heston(current_price: float, annualized_drift: float, initial_variance: float,
                  kappa: float, theta: float, sigma: float, rho: float,
                  time_horizon: int, num_paths: int) -> np.ndarray:
    """
    Vectorized implementation of Heston model simulation with GPU acceleration if available.
    
    Args:
        current_price: Current stock price
        annualized_drift: Annualized drift (mu)
        initial_variance: Initial variance (v0)
        kappa: Rate of mean reversion
        theta: Long-term variance
        sigma: Volatility of volatility
        rho: Correlation between price and volatility processes
        time_horizon: Number of days to simulate
        num_paths: Number of simulation paths
        
    Returns:
        Array of final prices for each path
    """
    dt = 1.0 / 252
    sqrt_dt = np.sqrt(dt)
    
    # Use GPU acceleration for large simulations
    if DEVICE.type != 'cpu' and num_paths * time_horizon > 5000:
        try:
            # Convert all parameters to tensors with the right dtype
            current_price_t = torch.tensor(current_price, device=DEVICE, dtype=TORCH_DTYPE)
            annualized_drift_t = torch.tensor(annualized_drift, device=DEVICE, dtype=TORCH_DTYPE)
            initial_variance_t = torch.tensor(initial_variance, device=DEVICE, dtype=TORCH_DTYPE)
            kappa_t = torch.tensor(kappa, device=DEVICE, dtype=TORCH_DTYPE)
            theta_t = torch.tensor(theta, device=DEVICE, dtype=TORCH_DTYPE)
            sigma_t = torch.tensor(sigma, device=DEVICE, dtype=TORCH_DTYPE)
            rho_t = torch.tensor(rho, device=DEVICE, dtype=TORCH_DTYPE)
            dt_t = torch.tensor(dt, device=DEVICE, dtype=TORCH_DTYPE)
            sqrt_dt_t = torch.tensor(sqrt_dt, device=DEVICE, dtype=TORCH_DTYPE)
            
            # Create tensors on GPU
            prices = torch.full((num_paths, time_horizon + 1), current_price, 
                               device=DEVICE, dtype=TORCH_DTYPE)
            variances = torch.full((num_paths, time_horizon + 1), initial_variance, 
                                  device=DEVICE, dtype=TORCH_DTYPE)
            
            # Generate random numbers on GPU
            z1 = torch.randn(num_paths, time_horizon, device=DEVICE, dtype=TORCH_DTYPE)
            z2 = torch.randn(num_paths, time_horizon, device=DEVICE, dtype=TORCH_DTYPE)
            
            # Calculate correlated random variables
            # Avoid the problematic calculation with explicit dtype
            one_minus_rho_squared = torch.tensor(1.0, device=DEVICE, dtype=TORCH_DTYPE) - rho_t * rho_t
            z2_correlated = rho_t * z1 + torch.sqrt(one_minus_rho_squared) * z2
            
            # Vectorized simulation step
            for t in range(time_horizon):
                # Ensure positive variance (GPU version)
                variances[:, t] = torch.clamp(variances[:, t], min=1e-8)
                
                # Price update
                price_drift = (annualized_drift_t - 0.5 * variances[:, t]) * dt_t
                price_diffusion = torch.sqrt(variances[:, t]) * sqrt_dt_t * z1[:, t]
                prices[:, t+1] = prices[:, t] * torch.exp(price_drift + price_diffusion)
                
                # Variance update
                var_drift = kappa_t * (theta_t - variances[:, t]) * dt_t
                var_diffusion = sigma_t * torch.sqrt(variances[:, t]) * sqrt_dt_t * z2_correlated[:, t]
                variances[:, t+1] = variances[:, t] + var_drift + var_diffusion
            
            # Return final prices as numpy array
            return prices[:, -1].cpu().numpy()
        except Exception as e:
            # Fallback to CPU if GPU fails
            logger.warning(f"GPU Heston simulation failed, falling back to CPU: {e}")
    
    # CPU implementation for smaller simulations or as fallback
    prices = np.full((num_paths, time_horizon + 1), current_price)
    variances = np.full((num_paths, time_horizon + 1), initial_variance)
    
    # Generate correlated random variables
    z1 = np.random.normal(0, 1, size=(num_paths, time_horizon))
    z2 = np.random.normal(0, 1, size=(num_paths, time_horizon))
    z2_correlated = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    # Vectorized simulation step
    for t in range(time_horizon):
        # Ensure positive variance
        variances[:, t] = np.maximum(variances[:, t], 1e-8)
        
        # Price update
        price_drift = (annualized_drift - 0.5 * variances[:, t]) * dt
        price_diffusion = np.sqrt(variances[:, t]) * sqrt_dt * z1[:, t]
        prices[:, t+1] = prices[:, t] * np.exp(price_drift + price_diffusion)
        
        # Variance update
        var_drift = kappa * (theta - variances[:, t]) * dt
        var_diffusion = sigma * np.sqrt(variances[:, t]) * sqrt_dt * z2_correlated[:, t]
        variances[:, t+1] = variances[:, t] + var_drift + var_diffusion
    
    return prices[:, -1]

# ============== SABR-CGMY Model ==============

def simulate_sabr_cgmy(current_price: float, initial_volatility: float, risk_free_rate: float,
                     sabr_alpha: float, sabr_beta: float, sabr_rho: float,
                     cgm_C: float, cgm_G: float, cgm_M: float, cgm_Y: float,
                     time_horizon: int, num_paths: int) -> np.ndarray:
    """
    Optimized implementation of SABR-CGMY model simulation.
    
    Args:
        current_price: Current stock price
        initial_volatility: Initial volatility (sigma0)
        risk_free_rate: Risk-free rate
        sabr_alpha: SABR volatility of volatility parameter
        sabr_beta: SABR beta parameter
        sabr_rho: SABR correlation parameter
        cgm_C: CGMY C parameter
        cgm_G: CGMY G parameter
        cgm_M: CGMY M parameter
        cgm_Y: CGMY Y parameter
        time_horizon: Number of days to simulate
        num_paths: Number of simulation paths
        
    Returns:
        Array of final prices for each path
    """
    dt = 1.0 / 252
    sqrt_dt = np.sqrt(dt)
    
    # Use GPU acceleration for large simulations
    if DEVICE.type != 'cpu' and num_paths * time_horizon > 5000:
        try:
            # Convert all parameters to tensors with the right dtype
            current_price_t = torch.tensor(current_price, device=DEVICE, dtype=TORCH_DTYPE)
            initial_volatility_t = torch.tensor(initial_volatility, device=DEVICE, dtype=TORCH_DTYPE)
            risk_free_rate_t = torch.tensor(risk_free_rate, device=DEVICE, dtype=TORCH_DTYPE)
            sabr_alpha_t = torch.tensor(sabr_alpha, device=DEVICE, dtype=TORCH_DTYPE)
            sabr_beta_t = torch.tensor(sabr_beta, device=DEVICE, dtype=TORCH_DTYPE)
            sabr_rho_t = torch.tensor(sabr_rho, device=DEVICE, dtype=TORCH_DTYPE)
            dt_t = torch.tensor(dt, device=DEVICE, dtype=TORCH_DTYPE)
            sqrt_dt_t = torch.tensor(sqrt_dt, device=DEVICE, dtype=TORCH_DTYPE)
            
            # Pre-allocate tensors on GPU
            S = torch.full((num_paths, time_horizon + 1), current_price, 
                          device=DEVICE, dtype=TORCH_DTYPE)
            sigma_arr = torch.full((num_paths, time_horizon + 1), initial_volatility, 
                                  device=DEVICE, dtype=TORCH_DTYPE)
            
            # Pre-generate random variables on GPU
            dW1 = torch.randn(num_paths, time_horizon, device=DEVICE, dtype=TORCH_DTYPE) * sqrt_dt
            dW2_uncorr = torch.randn(num_paths, time_horizon, device=DEVICE, dtype=TORCH_DTYPE) * sqrt_dt
            
            # Calculate correlated random variables
            one_minus_rho_squared = torch.tensor(1.0, device=DEVICE, dtype=TORCH_DTYPE) - sabr_rho_t * sabr_rho_t
            dW2 = sabr_rho_t * dW1 + torch.sqrt(one_minus_rho_squared) * dW2_uncorr
            
            # Pre-compute CGMY increments (on CPU since it's complex)
            cgmy_increments_np = simulate_cgmy_increments_optimized(cgm_C, cgm_G, cgm_M, cgm_Y, dt, num_paths * time_horizon)
            cgmy_increments_np = cgmy_increments_np.reshape(num_paths, time_horizon)
            cgmy_increments = torch.tensor(cgmy_increments_np, device=DEVICE, dtype=TORCH_DTYPE)
            
            # Half volatility squared * dt term for efficiency
            half_alpha_squared_dt = torch.tensor(0.5, device=DEVICE, dtype=TORCH_DTYPE) * sabr_alpha_t**2 * dt_t
            
            # Simulation loop
            for t in range(time_horizon):
                # Update volatility
                sigma_arr[:, t+1] = sigma_arr[:, t] * torch.exp(
                    sabr_alpha_t * dW2[:, t] - half_alpha_squared_dt
                )
                
                # Calculate SABR component
                S_pow_beta = torch.pow(S[:, t], sabr_beta_t)
                sabr_component = sigma_arr[:, t] * S_pow_beta * dW1[:, t]
                
                # Update price
                S[:, t+1] = S[:, t] * torch.exp(
                    risk_free_rate_t * dt_t + sabr_component / S[:, t] + cgmy_increments[:, t]
                )
            
            return S[:, -1].cpu().numpy()
        except Exception as e:
            # Fallback to CPU if GPU fails
            logger.warning(f"GPU SABR-CGMY simulation failed, falling back to CPU: {e}")
    
    # CPU implementation for smaller simulations or as fallback
    S = np.full((num_paths, time_horizon + 1), current_price)
    sigma_arr = np.full((num_paths, time_horizon + 1), initial_volatility)
    
    # Pre-generate correlated random variables
    dW1 = np.random.normal(0, 1, size=(num_paths, time_horizon)) * sqrt_dt
    dW2_uncorr = np.random.normal(0, 1, size=(num_paths, time_horizon)) * sqrt_dt
    dW2 = sabr_rho * dW1 + np.sqrt(1 - sabr_rho**2) * dW2_uncorr
    
    # Pre-compute CGMY increments
    cgmy_increments = simulate_cgmy_increments_optimized(cgm_C, cgm_G, cgm_M, cgm_Y, dt, num_paths * time_horizon)
    cgmy_increments = cgmy_increments.reshape(num_paths, time_horizon)
    
    # Simulation loop with vectorized operations
    half_alpha_squared_dt = 0.5 * sabr_alpha**2 * dt
    for t in range(time_horizon):
        # Update volatility
        sigma_arr[:, t+1] = sigma_arr[:, t] * np.exp(
            sabr_alpha * dW2[:, t] - half_alpha_squared_dt
        )
        
        # Update price
        S_pow_beta = np.power(S[:, t], sabr_beta)
        sabr_component = sigma_arr[:, t] * S_pow_beta * dW1[:, t]
        S[:, t+1] = S[:, t] * np.exp(
            risk_free_rate * dt + sabr_component / S[:, t] + cgmy_increments[:, t]
        )
    
    return S[:, -1]