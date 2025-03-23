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

# ============== CGMY Implementation ==============

def _cgmy_levy_measure(x, C, G, M, Y):
    """
    Calculate the CGMY Lévy measure for a given x value.
    
    Args:
        x: Point to evaluate
        C: CGMY C parameter (scale)
        G: CGMY G parameter (negative jumps)
        M: CGMY M parameter (positive jumps)
        Y: CGMY Y parameter (jump activity)
        
    Returns:
        Value of the Lévy measure at x
    """
    if x == 0:
        return 0
    elif x < 0:
        return C * np.exp(G * x) * np.power(np.abs(x), -1 - Y)
    else:
        return C * np.exp(-M * x) * np.power(x, -1 - Y)

def _cgmy_characteristic_function(u, C, G, M, Y, t):
    """
    Calculate the characteristic function of the CGMY process.
    
    Args:
        u: Frequency parameter
        C: CGMY C parameter
        G: CGMY G parameter
        M: CGMY M parameter
        Y: CGMY Y parameter
        t: Time parameter
        
    Returns:
        Characteristic function value
    """
    # Ensure stability
    G_adj = max(G, 1e-8)
    M_adj = max(M, 1e-8)
    
    # Calculate characteristic function
    omega = C * gamma(-Y) * (
        np.power(M_adj, Y) - np.power(M_adj - 1j * u, Y) +
        np.power(G_adj, Y) - np.power(G_adj + 1j * u, Y)
    )
    
    return np.exp(t * omega)

@lru_cache(maxsize=32)
def _get_cgmy_params_hash(C, G, M, Y, dt):
    """
    Create a hash of the CGMY parameters for caching.
    
    Args:
        C, G, M, Y: CGMY parameters
        dt: Time step
        
    Returns:
        Tuple hash of parameters
    """
    # Round parameters for cache stability
    C_r = round(C, 6)
    G_r = round(G, 6)
    M_r = round(M, 6)
    Y_r = round(Y, 6)
    dt_r = round(dt, 10)
    
    return (C_r, G_r, M_r, Y_r, dt_r)

def _simulate_cgmy_increments_small_jumps(C, G, M, Y, dt, num_increments):
    """
    Simulate the small jumps component of the CGMY process using Gaussian approximation.
    
    Args:
        C, G, M, Y: CGMY parameters
        dt: Time step
        num_increments: Number of increments to generate
        
    Returns:
        Array of small jump increments
    """
    # Truncation level for small jumps
    epsilon = 0.1
    
    # Calculate variance of small jumps
    var_small_jumps = C * dt * (
        gamma(2 - Y) * (np.power(epsilon, 2 - Y) / (2 - Y)) * 
        (1 / np.power(G, 2-Y) + 1 / np.power(M, 2-Y))
    )
    
    # Generate Gaussian approximation
    return np.random.normal(0, np.sqrt(var_small_jumps), num_increments)

def _simulate_cgmy_increments_large_jumps(C, G, M, Y, dt, num_increments):
    """
    Simulate the large jumps component of the CGMY process.
    
    Args:
        C, G, M, Y: CGMY parameters
        dt: Time step
        num_increments: Number of increments to generate
        
    Returns:
        Array of large jump increments
    """
    # Truncation level for large jumps
    epsilon = 0.1
    
    # Calculate intensity of positive and negative large jumps
    lambda_pos = C * dt * epsilon**(-Y) / Y * np.exp(-M * epsilon)
    lambda_neg = C * dt * epsilon**(-Y) / Y * np.exp(-G * epsilon)
    
    # Expected number of jumps
    mean_num_jumps = lambda_pos + lambda_neg
    
    # Generate actual number of jumps
    num_jumps = np.random.poisson(mean_num_jumps)
    
    if num_jumps == 0:
        return np.zeros(num_increments)
    
    # Generate jump sizes
    jumps = np.zeros(num_jumps)
    
    # Probability of positive jump
    prob_pos = lambda_pos / (lambda_pos + lambda_neg)
    
    # Generate jump directions
    directions = np.random.binomial(1, prob_pos, num_jumps)
    
    # Generate jump sizes based on direction
    for i in range(num_jumps):
        if directions[i] == 1:  # Positive jump
            # Sample from truncated exponential
            u = np.random.uniform(0, 1)
            jumps[i] = epsilon - (1/M) * np.log(u * (1 - np.exp(-M * (10 - epsilon))) + np.exp(-M * (10 - epsilon)))
        else:  # Negative jump
            # Sample from truncated exponential
            u = np.random.uniform(0, 1)
            jumps[i] = -(epsilon - (1/G) * np.log(u * (1 - np.exp(-G * (10 - epsilon))) + np.exp(-G * (10 - epsilon))))
    
    # Distribute jumps across increments
    result = np.zeros(num_increments)
    jump_positions = np.random.randint(0, num_increments, num_jumps)
    
    # Add jumps at their positions
    for i, pos in enumerate(jump_positions):
        result[pos] += jumps[i]
    
    return result

def _compensate_jumps(C, G, M, Y, dt):
    """
    Calculate the compensation term for the CGMY process to ensure martingality.
    
    Args:
        C, G, M, Y: CGMY parameters
        dt: Time step
        
    Returns:
        Compensation term
    """
    # Martingale compensation term
    compensation = 0.0
    
    # If Y is close to 0 or 1, use special cases to avoid numerical issues
    if abs(Y - 0.0) < 1e-6:
        compensation = C * dt * (np.log(G / (G - 1)) + np.log(M / (M + 1)))
    elif abs(Y - 1.0) < 1e-6:
        compensation = C * dt * (
            G * np.log(G) - (G - 1) * np.log(G - 1) -
            M * np.log(M) + (M + 1) * np.log(M + 1)
        )
    else:
        # Standard formula
        compensation = C * dt * gamma(-Y) * (
            np.power(M, Y) * (1 - Y) - M * np.power(M - 1, Y - 1) +
            np.power(G, Y) * (1 - Y) + G * np.power(G + 1, Y - 1)
        )
    
    return compensation

def simulate_cgmy_increments_optimized(C, G, M, Y, dt, num_increments):
    """
    Optimized implementation to simulate CGMY process increments.
    Uses caching for efficiency and different methods based on parameters.
    
    Args:
        C: CGMY C parameter (scale)
        G: CGMY G parameter (negative jumps)
        M: CGMY M parameter (positive jumps)
        Y: CGMY Y parameter (jump activity)
        dt: Time step size
        num_increments: Number of increments to simulate
        
    Returns:
        Array of CGMY increments
    """
    # Ensure parameters are in valid ranges
    C = max(1e-8, C)
    G = max(1e-8, G)
    M = max(1e-8, M)
    Y = min(max(0.01, Y), 1.99)  # Y must be in (0, 2)
    
    # Create parameter hash for cache lookup
    param_hash = _get_cgmy_params_hash(C, G, M, Y, dt)
    
    # Check if compensation term is in cache
    if param_hash in CGMY_CACHE:
        compensation = CGMY_CACHE[param_hash]
    else:
        # Calculate compensation term and cache it
        compensation = _compensate_jumps(C, G, M, Y, dt)
        CGMY_CACHE[param_hash] = compensation
    
    # Different simulation approaches based on Y parameter
    if Y < 0.5:
        # For small Y, compound Poisson process approximation works well
        small_jumps = np.zeros(num_increments)
        large_jumps = _simulate_cgmy_increments_large_jumps(C, G, M, Y, dt, num_increments)
        increments = small_jumps + large_jumps
    elif Y >= 0.5 and Y < 1.5:
        # For medium Y, use combination of small and large jumps
        small_jumps = _simulate_cgmy_increments_small_jumps(C, G, M, Y, dt, num_increments)
        large_jumps = _simulate_cgmy_increments_large_jumps(C, G, M, Y, dt, num_increments)
        increments = small_jumps + large_jumps
    else:
        # For large Y, stable approximation works better
        # Calculate parameters for alpha-stable approximation
        alpha = Y
        beta = (G - M) / (G + M)
        sigma = np.power(C * dt * gamma(2 - Y) * np.cos(np.pi * Y / 2) * 
                        np.power(G + M, Y), 1/alpha)
        mu = 0.0
        
        # Generate stable random variables
        increments = levy_stable.rvs(
            alpha=alpha, beta=beta, loc=mu, scale=sigma, size=num_increments
        )
    
    # Apply compensation term for martingality
    increments = increments - compensation
    
    return increments

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