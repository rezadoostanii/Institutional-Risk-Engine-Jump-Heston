# 🏛 Institutional Double Heston Risk Engine

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Model](https://img.shields.io/badge/Model-Double%20Heston%20%2B%20Jumps-red)
![Risk](https://img.shields.io/badge/Risk-VaR%20%2F%20CVaR-orange)
![Backtesting](https://img.shields.io/badge/Backtesting-Kupiec%20%2F%20Christoffersen-green)

Advanced quantitative risk engine implementing a **Double Heston stochastic volatility model with jumps, t-distributed shocks, and fractional volatility effects**, designed for institutional-grade market risk modeling and regulatory validation.

---

## 🚀 Key Features

- Double Heston stochastic volatility model  
- Jump diffusion (Poisson + Gaussian jumps)  
- Student-t heavy-tailed innovations  
- Fractional volatility (Hurst exponent via variogram)  
- VaR & CVaR (95% / 99%)  
- Basel traffic-light classification  
- Kupiec & Christoffersen backtests  
- Realized variance / bipower variation jump detection  
- Tail risk diagnostics (skew, kurtosis, drawdown)  
- Rolling VaR backtesting framework  
- Multi-horizon risk forecasting (1–10 days)  
- Monte Carlo simulation engine  

---

## 📊 Assets Modeled

| Asset | Type | Regime |
|------|------|--------|
| NVDA | Single Stock | High Volatility / Jump-Dominant |
| GSPC | Index | Low Volatility / Diffusion-Dominant |

---

## 🧠 Model Architecture

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

---

## ⚙️ Methodology

### 1. Stochastic Volatility Engine
- Double Heston variance process  
- Mean-reverting volatility factors  
- Correlated Brownian motions  
- Andersen Quadratic Exponential discretization scheme  

### 2. Jump & Tail Modeling
- Poisson jump arrivals  
- Gaussian jump size distribution  
- Student-t innovations (fat tails)  

### 3. Memory Effects
- Fractional volatility via Hurst exponent  
- Long-memory correction in volatility scaling  

### 4. Monte Carlo Simulation Engine
- High-dimensional path generation  
- Numba-accelerated computation  
- Multi-horizon risk projection  

---

## 📈 Empirical Results

### 🧠 NVIDIA (NVDA) — High Volatility / Jump-Dominant

| Metric | Value |
|------|------|
| t-dof (ν) | 11.98 |
| Jump intensity (λ) | 0.158 |
| Hurst exponent (H) | 0.050 |
| Kupiec test | PASS |
| Christoffersen test | PASS |
| Basel status | 🟡 YELLOW |

**Interpretation:**
- Strong jump behavior detected  
- Heavy tails captured effectively  
- Model performs robustly under stress regimes  

---

### 📉 S&P 500 (GSPC) — Diffusion-Dominant

| Metric | Value |
|------|------|
| t-dof (ν) | 8.25 |
| Jump intensity (λ) | ~0.000 |
| Hurst exponent (H) | 0.062 |
| Kupiec test | FAIL |
| Christoffersen test | INCONCLUSIVE |
| Basel status | 🟢 GREEN |

**Interpretation:**
- Low jump frequency regime  
- Diffusion dominates dynamics  
- Model shows over-parameterization in index setting  

---

## 🏦 Regulatory Backtesting

- Kupiec Proportion of Failures Test  
- Christoffersen Independence Test  
- Basel Traffic Light System  

| Status | Meaning |
|--------|--------|
| 🟢 Green | Acceptable model performance |
| 🟡 Yellow | Monitor / borderline risk |
| 🔴 Red | Model rejection |

---

## 📉 Risk Metrics

- Value-at-Risk (VaR 95%, 99%)  
- Conditional VaR (CVaR)  
- Max Drawdown  
- Tail Ratio  
- Downside Volatility  
- Kolmogorov–Smirnov test  
- Jarque–Bera test  

---

## 📌 Key Insight

> Risk models are **asset-dependent, not universal**

Empirical evidence shows strong regime sensitivity:

- Single-name equities → jump-driven dynamics dominate  
- Index-level assets → diffusion-dominant behavior prevails  

This implies model selection depends on **market microstructure**, not only statistical fit.

---

## ⚠️ Limitations

- Sensitive to rolling window calibration  
- Jump detection noise in low-vol regimes  
- Fractional volatility is empirical (not structural)  
- Parameter drift over time  

---

## 🧰 Technology Stack

- Python 3.12  
- NumPy / Pandas  
- SciPy / Statsmodels  
- Matplotlib  
- Numba (JIT acceleration)  
- SQLite (data backend)  

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/rezadoostanii/Institutional-Risk-Engine-Jump-Heston.git
cd Institutional-Risk-Engine-Jump-Heston

# Install dependencies
pip install -r requirements.txt

# Run the engine (compares NVDA vs GSPC)
python main.py

python main.py --compare
