This project presents an institutional-style quantitative risk engine designed for advanced market risk modeling under non-Gaussian dynamics.

The framework integrates:

Double Heston stochastic volatility
Jump diffusion processes
Student-t heavy-tailed innovations
Fractional volatility (long-memory effects)

The objective is to evaluate whether enriched stochastic models improve risk estimation accuracy compared to classical Gaussian-based frameworks.

Key Insight

Risk models are asset-dependent, not universal.

Empirical results show strong regime sensitivity:

Single-name equities → jump-driven dynamics dominate
Index-level assets → diffusion-dominant behavior prevails

This implies model selection is a function of market microstructure, not only statistical fit.

Model Architecture
Market Data
   ↓
Parameter Calibration
   ↓
Double Heston Volatility Process
   ↓
Jump Diffusion Layer
   ↓
Fractional Volatility Adjustment
   ↓
Monte Carlo Simulation Engine
   ↓
Risk Metrics (VaR / CVaR)
   ↓
Regulatory Backtesting
   ↓
Diagnostic Validation
Methodology
1. Stochastic Volatility Engine
Double Heston variance process
Mean-reverting stochastic volatility factors
Correlated Brownian motions
Andersen Quadratic Exponential discretization
2. Jump & Tail Modeling
Poisson jump arrivals
Gaussian jump size distribution
Student-t innovations (heavy tails)
3. Memory Effects
Fractional volatility via Hurst exponent
Long-memory correction in volatility scaling
4. Monte Carlo Engine
High-dimensional path generation
Numba-accelerated computation
Multi-horizon risk projection
📊 Empirical Results
🧠 NVIDIA (NVDA) — High Volatility / Jump-Dominant
Metric\tValue
t-dof (ν)\t11.98
Jump intensity (λ)\t0.158
Hurst (H)\t0.050
Kupiec Test\tPASS
Christoffersen\tPASS
Basel Status\t🟡 YELLOW

Interpretation:

Strong jump behavior
Heavy tails captured well
Model performs reliably under stress
📉 S&P 500 (GSPC) — Diffusion-Dominant
Metric\tValue
t-dof (ν)\t8.25
Jump intensity (λ)\t~0.000
Hurst (H)\t0.062
Kupiec Test\tFAIL
Christoffersen\tINCONCLUSIVE
Basel Status\t🟢 GREEN

Interpretation:

Low jump frequency
Diffusion-only structure dominates
Model becomes over-parameterized
🏦 Regulatory Backtesting
Kupiec Proportion of Failures Test
Christoffersen Independence Test
Basel Traffic Light System:
Status\tMeaning
🟢 Green\tAcceptable model performance
🟡 Yellow\tMonitor / borderline
🔴 Red\tModel rejection
📉 Risk Metrics
Value-at-Risk (VaR 95%, 99%)
Conditional VaR (CVaR)
Max Drawdown
Tail Ratio
Downside Volatility
Kolmogorov–Smirnov test
Jarque–Bera test
📌 Key Finding

Stochastic volatility + jumps improve risk estimation for high-volatility equities,
but may lead to misspecification in index-level assets.

⚠️ Limitations
Sensitivity to rolling window calibration
Jump detection noise in low-vol regimes
Empirical (not structural) fractional volatility
Non-stationary parameter drift
🧰 Technology Stack
Python 3.12
NumPy / Pandas
SciPy / Statsmodels
Matplotlib
Numba (JIT acceleration)
SQLite (data backend)
🚀 Future Enhancements
Regime-switching volatility models
Portfolio-level multivariate extension
GPU-accelerated Monte Carlo
Neural SDE-based volatility modeling
Real-time risk engine deployment
⚡ Quick Start
git clone https://github.com/rezadoostanii/risk-engine.git
cd risk-engine

pip install -r requirements.txt
python main.py --compare
🎯 Final Note

This project is designed as an institutional research prototype, demonstrating:

Stochastic volatility modeling
Tail risk quantification
Regulatory backtesting
Monte Carlo risk simulation"""