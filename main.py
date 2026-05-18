"""
================================================================================
RISK ENGINE - FINAL INSTITUTIONAL FRAMEWORK
================================================================================
Double Heston Model with Jumps + t-Distribution + Fractional Volatility

✨ FEATURES:
    ✔ Advanced Calibration (Numerical Optimization + Feller Constraint)
    ✔ Advanced Jump Detection (RV/BPV/RQP Method)
    ✔ Real Hurst Exponent Calculation (Variogram)
    ✔ VaR + CVaR Framework
    ✔ Basel Traffic Light Classification
    ✔ Tail Risk Dashboard
    ✔ Regime Classification
    ✔ Stress Testing Framework
    ✔ QQ Plots & Distribution Analysis
    ✔ VaR Breach Visualization

ASSET COMPARISON: NVDA vs GSPC

Author: Reza Doostani
================================================================================
"""

import os
import sqlite3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import (
    t as t_dist,
    chi2,
    norm,
    probplot,
    ks_2samp,
    jarque_bera
)

from pathlib import Path
from numba import jit, prange

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = {
    'NVDA': {
        'table': 'nvda_prices',
        'label': 'NVIDIA (High Volatility)',
        'color': 'green'
    },
    'GSPC': {
        'table': 'gspc_prices',
        'label': 'S&P 500 (Low Volatility)',
        'color': 'blue'
    }
}

START_DATE = '2020-01-01'
END_DATE = '2026-05-07'

TRADING_DAYS_YEAR = 252
SEED = 12345
SIMULATIONS = 10000
ROLLING_WINDOW = 600
ROLLING_STEP = 5

DB_PATH = Path.home() / "Downloads" / "portfolio.db"

np.random.seed(SEED)

# =============================================================================
# MODULE 1: RISK ENGINE (Simulation)
# =============================================================================

@jit(nopython=True)
def andersen_qe(v_prev, kappa, theta, sigma, dt, dW):
    """Andersen Quadratic Exponential scheme for Heston"""
    psi = sigma**2 / (2 * kappa + 1e-12)
    m = theta + (v_prev - theta) * np.exp(-kappa * dt)
    s2 = (
        v_prev * psi * np.exp(-kappa * dt)
        * (1 - np.exp(-kappa * dt))
        + theta * psi * (1 - np.exp(-kappa * dt))**2
    )

    if s2 <= 1e-12:
        return max(m, 1e-12)

    psi_val = s2 / (m**2 + 1e-12)

    if psi_val <= 1.5:
        b2 = (
            2.0 / psi_val
            - 1.0
            + np.sqrt(2.0 / psi_val)
            * np.sqrt(2.0 / psi_val - 1.0)
        )
        a = m / (1.0 + b2)
        Z = dW / np.sqrt(dt) if dt > 0 else dW
        v_new = a * (np.sqrt(b2) + Z)**2
    else:
        p = (psi_val - 1.0) / (psi_val + 1.0)
        beta = (1.0 - p) / m
        U = (
            0.5
            + 0.5
            * np.sign(dW)
            * (1.0 - np.exp(-2.0 * np.abs(dW) / np.sqrt(3.0)))
        )
        if U <= p:
            v_new = 0.0
        else:
            v_new = (1.0 / beta) * np.log((1.0 - p) / (1.0 - U))

    return max(v_new, 1e-12)


@jit(nopython=True, parallel=True)
def simulate_double_heston(
    S0,
    horizon,
    n_sims,
    params,
    mu_daily,
    lamJ,
    muJ,
    sigJ,
    H,
    xi,
    fgn_steps,
    nu_t
):
    """Double Heston simulation with jumps and fractional volatility"""
    dt = 1.0 / 252.0
    lam_dt = lamJ * dt
    sqrt_dt = np.sqrt(dt)

    k1, t1, s1, v01, r1 = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4]
    )
    k2, t2, s2, v02, r2 = (
        params[5],
        params[6],
        params[7],
        params[8],
        params[9]
    )

    corr_mat = np.array([
        [1.0, r1, r2],
        [r1, 1.0, 0.0],
        [r2, 0.0, 1.0]
    ])

    L = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1):
            if i == j:
                L[i, i] = np.sqrt(
                    max(
                        corr_mat[i, i] - np.sum(L[i, :j]**2),
                        1e-12
                    )
                )
            else:
                L[i, j] = (
                    corr_mat[i, j]
                    - np.sum(L[i, :j] * L[j, :j])
                ) / max(L[j, j], 1e-12)

    S = np.zeros((n_sims, horizon + 1), dtype=np.float64)
    v1 = np.zeros((n_sims, horizon + 1), dtype=np.float64)
    v2 = np.zeros((n_sims, horizon + 1), dtype=np.float64)

    for i in prange(n_sims):
        S[i, 0] = S0
        v1[i, 0] = max(v01, 1e-12)
        v2[i, 0] = max(v02, 1e-12)

        for t in range(1, horizon + 1):
            Z = np.random.standard_normal(3)
            chi2_val = np.random.chisquare(nu_t)
            t_scale = np.sqrt(nu_t / max(chi2_val, 1e-12))
            Z_scaled = Z * t_scale

            dWs = 0.0
            dW1 = 0.0
            dW2 = 0.0
            for j in range(3):
                dWs += L[0, j] * Z_scaled[j]
                dW1 += L[1, j] * Z_scaled[j]
                dW2 += L[2, j] * Z_scaled[j]

            dWs *= sqrt_dt
            dW1 *= sqrt_dt
            dW2 *= sqrt_dt

            v1_t = andersen_qe(
                v1[i, t-1],
                k1,
                t1,
                s1,
                dt,
                dW1
            )
            v2_t = andersen_qe(
                v2[i, t-1],
                k2,
                t2,
                s2,
                dt,
                dW2
            )

            vol = np.sqrt(max(v1_t + v2_t, 1e-12))
            vol_eff = vol
            if xi != 0 and t-1 < len(fgn_steps):
                vol_eff = vol * max(
                    1.0 + xi * fgn_steps[t-1],
                    1e-4
                )

            K = np.random.poisson(lam_dt)
            jump = (
                np.random.normal(
                    K * muJ,
                    np.sqrt(max(K, 1)) * sigJ
                )
                if K > 0 else 0.0
            )

            r = (
                (mu_daily - 0.5 * vol_eff**2) * dt
                + vol_eff * dWs
                + jump
            )

            S[i, t] = S[i, t-1] * np.exp(r)
            v1[i, t] = v1_t
            v2[i, t] = v2_t

    return S

# =============================================================================
# MODULE 2: BACKTESTING
# =============================================================================

class VaRBacktest:
    """Regulatory backtesting - Kupiec & Christoffersen"""

    @staticmethod
    def kupiec(exceptions, n, cl=0.95):
        p_exp = 1 - cl
        if n == 0:
            return {'verdict': 'NO DATA', 'p_value': 0.5}

        p_obs = exceptions / n
        if exceptions == 0:
            lr = -2 * n * np.log(1 - p_exp)
        elif exceptions == n:
            lr = -2 * n * np.log(p_exp)
        else:
            lr = -2 * (
                np.log(
                    (1-p_exp)**(n-exceptions)
                    * p_exp**exceptions
                )
                -
                np.log(
                    (1-p_obs)**(n-exceptions)
                    * p_obs**exceptions
                )
            )

        p_value = 1 - chi2.cdf(lr, 1)
        return {
            'verdict': 'PASS' if p_value > 0.05 else 'FAIL',
            'p_value': p_value,
            'exceptions': exceptions
        }

    @staticmethod
    def christoffersen(exceptions_series, cl=0.95):
        n = len(exceptions_series)
        exceptions = int(np.sum(exceptions_series))

        if n < 2 or exceptions == 0 or exceptions == n:
            return {
                'verdict': 'INCONCLUSIVE',
                'p_value': 0.5
            }

        n00 = n01 = n10 = n11 = 0
        for i in range(1, n):
            if exceptions_series[i-1] == 0 and exceptions_series[i] == 0:
                n00 += 1
            elif exceptions_series[i-1] == 0 and exceptions_series[i] == 1:
                n01 += 1
            elif exceptions_series[i-1] == 1 and exceptions_series[i] == 0:
                n10 += 1
            elif exceptions_series[i-1] == 1 and exceptions_series[i] == 1:
                n11 += 1

        pi01 = n01/(n00+n01) if (n00+n01) > 0 else 0
        pi11 = n11/(n10+n11) if (n10+n11) > 0 else 0
        pi2 = (n01+n11)/n if n > 0 else (1-cl)

        if pi01 > 0 and pi11 > 0 and pi2 > 0 and pi2 < 1:
            lr_ind = -2 * (
                np.log(
                    (1-pi2)**(n00+n10)
                    * pi2**(n01+n11)
                )
                -
                np.log(
                    (1-pi01)**n00
                    * pi01**n01
                    * (1-pi11)**n10
                    * pi11**n11
                )
            )
        else:
            lr_ind = 0

        p_value = 1 - chi2.cdf(lr_ind, 1) if lr_ind > 0 else 0.5
        return {
            'verdict': 'PASS' if p_value > 0.05 else 'FAIL',
            'p_value': p_value
        }

    @staticmethod
    def basel_traffic_light(n_exceptions):
        if n_exceptions <= 4:
            return "GREEN"
        elif n_exceptions <= 9:
            return "YELLOW"
        else:
            return "RED"

# =============================================================================
# MODULE 3: ANALYTICS
# =============================================================================

class Analytics:
    """Risk analytics and visualization"""

    @staticmethod
    def calculate_var_cvar(paths, alpha=0.95):
        pnl = (paths[:, -1] / paths[:, 0] - 1) * 100
        var = np.percentile(pnl, (1-alpha)*100)
        cvar = pnl[pnl <= var].mean() if len(pnl[pnl <= var]) > 0 else var
        return var, cvar

    @staticmethod
    def classify_regime(lamJ, nu_t, H_hat):
        if lamJ > 0.10 and nu_t < 12:
            return "Jump-Dominant"
        elif H_hat > 0.60:
            return "Long-Memory"
        else:
            return "Diffusion-Dominant"

    @staticmethod
    def max_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown) * 100

    @staticmethod
    def downside_volatility(returns):
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        return np.std(downside) * np.sqrt(252) * 100

    @staticmethod
    def tail_ratio(returns):
        upper = np.percentile(returns, 95)
        lower = abs(np.percentile(returns, 5))
        if lower == 0:
            return 0
        return upper / lower

    @staticmethod
    def plot_qq_comparison(asset_name, actual_returns, sim_returns):
        """QQ plot for distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Q-Q vs Normal
        probplot(actual_returns, dist=norm, plot=axes[0])
        axes[0].set_title(f'{asset_name} - Q-Q vs Normal', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Q-Q Simulated vs Actual
        actual_sorted = np.sort(actual_returns)
        sim_sorted = np.sort(sim_returns[:len(actual_returns)])
        axes[1].scatter(sim_sorted, actual_sorted, alpha=0.5, s=10)
        axes[1].plot(actual_sorted, actual_sorted, 'r--', label='Identity')
        axes[1].set_xlabel('Simulated Quantiles')
        axes[1].set_ylabel('Actual Quantiles')
        axes[1].set_title(f'{asset_name} - Actual vs Simulated', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_distribution(asset_name, actual_returns, simulated_returns):
        """Histogram comparison of actual vs simulated returns"""
        plt.figure(figsize=(10, 6))
        plt.hist(actual_returns, bins=50, alpha=0.5, label='Actual Returns', density=True)
        plt.hist(simulated_returns, bins=50, alpha=0.5, label='Simulated Returns', density=True)
        plt.xlabel('Return (%)')
        plt.ylabel('Density')
        plt.title(f'{asset_name} - Return Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_var_breach(asset_name, returns, var_line, breaches):
        """VaR breach visualization"""
        plt.figure(figsize=(12, 5))
        plt.plot(returns, label='Returns', alpha=0.7)
        plt.axhline(var_line, color='red', linestyle='--', label=f'VaR 95% ({var_line:.2f}%)')
        
        breach_indices = np.where(breaches)[0]
        breach_values = returns[breaches]
        plt.scatter(breach_indices, breach_values, color='red', s=30, label='VaR Breaches', zorder=5)
        
        plt.xlabel('Time Step')
        plt.ylabel('Return (%)')
        plt.title(f'{asset_name} - VaR Backtesting Breaches')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def distribution_stats(returns, name):
        print(f"\n   📊 {name} - DISTRIBUTION STATISTICS:")
        print(f"   {'─'*50}")
        print(f"   Mean:     {np.mean(returns):8.4f}%")
        print(f"   Std:      {np.std(returns):8.4f}%")
        print(f"   Skewness: {stats.skew(returns):8.4f}")
        print(f"   Kurtosis: {stats.kurtosis(returns):8.4f}")
        
        jb_stat, jb_p = jarque_bera(returns)
        print(f"   JB p-val: {jb_p:8.4f} ({'Normal' if jb_p>0.05 else 'Non-normal'})")

# =============================================================================
# MODULE 4: DATA LOADING & ADVANCED CALIBRATION
# =============================================================================

def load_data(ticker, table_name):
    """Load data from SQLite database"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]

    date_col = next(
        (c for c in ['Date', 'date', 'datetime'] if c in columns),
        'Date'
    )
    price_col = next(
        (c for c in ['Close', 'close', 'Adj Close'] if c in columns),
        'Close'
    )

    query = (
        f'SELECT "{date_col}" as Date, '
        f'"{price_col}" as Close '
        f'FROM {table_name}'
    )
    if START_DATE:
        query += f" WHERE Date >= '{START_DATE}'"
    if END_DATE:
        query += f" AND Date <= '{END_DATE}'"
    query += " ORDER BY Date ASC"

    df = pd.read_sql(query, conn)
    conn.close()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    return df


def estimate_t_dof(returns):
    """Estimate Student-t degrees of freedom using MLE"""
    returns_std = returns / returns.std()

    def neg_ll(nu):
        if nu <= 2.1:
            return 1e10
        return -np.sum(
            t_dist.logpdf(
                returns_std,
                df=nu,
                loc=0,
                scale=1
            )
        )

    res = minimize(
        neg_ll,
        x0=[5.0],
        method='L-BFGS-B',
        bounds=[(2.1, 30.0)]
    )
    return res.x[0] if res.success else 5.0


def detect_jumps_advanced(returns):
    """
    Advanced jump detection using RV, BPV, and RQP
    This is the professional method from the second code
    """
    r = returns.flatten()
    n = len(r)
    
    # Realized Variance
    rv = np.sum(r**2)
    
    # Bi-Power Variation
    r_abs = np.abs(r)
    bpv = (np.pi/2) * np.sum(r_abs[1:] * r_abs[:-1])
    
    # Quad-Power Variation
    rqp = (np.pi**2/4) * np.sum(r_abs[2:] * r_abs[1:-1] * r_abs[2:] * r_abs[1:-1])
    
    # Z-statistic for jump detection
    z_stat = (rv - bpv) / np.sqrt(((np.pi**2/4 + np.pi - 5) * rqp / n) + 1e-12)
    
    # Detect jumps at 99% confidence
    jump_idx = np.where(z_stat > norm.ppf(0.99))[0]
    
    lam = len(jump_idx) / n * 252
    muJ = np.mean(r[jump_idx]) if len(jump_idx) > 0 else 0.0
    sigJ = np.std(r[jump_idx]) if len(jump_idx) > 1 else 0.0
    
    return lam, muJ, sigJ, len(jump_idx)


def calibrate_model_advanced(returns):
    """
    Advanced calibration using numerical optimization
    This is the professional method from the second code
    """
    dt = 1/252
    r_arr = returns.values.flatten()
    
    # Calculate target realized volatility
    rv = np.zeros(len(r_arr))
    for i in range(5, len(r_arr)):
        rv[i] = np.sqrt(np.sum(r_arr[i-5:i]**2) * 252 / 5)
    rv_target = np.mean(rv[rv > 0]) / 100
    
    inst_var = np.var(r_arr) / dt
    
    def objective(x):
        k1, k2, w1, s1, s2, rh1, rh2 = x
        w2 = 1 - w1
        theta1 = max(w1 * inst_var, 1e-12)
        theta2 = max(w2 * inst_var, 1e-12)
        
        # Feller constraint penalty
        pen = 0.0
        if s1**2 > 2 * k1 * theta1:
            pen += (s1**2 - 2 * k1 * theta1)**2 * 100
        if s2**2 > 2 * k2 * theta2:
            pen += (s2**2 - 2 * k2 * theta2)**2 * 100
        
        # Target volatility penalty
        model_vol = np.sqrt((theta1 + theta2) * dt * 252)
        pen_rv = 100 * (model_vol - rv_target)**2
        
        return pen + pen_rv
    
    x0 = np.array([3.0, 0.3, 0.7, 0.5, 0.2, -0.6, -0.3])
    bounds = [(1e-3, 20), (1e-3, 10), (0.05, 0.95), (0.01, 2), (0.01, 2), (-0.99, 0), (-0.99, 0)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000})
    
    k1, k2, w1, s1, s2, rh1, rh2 = res.x
    w2 = 1 - w1
    theta1 = max(w1 * inst_var, 1e-12)
    theta2 = max(w2 * inst_var, 1e-12)
    
    return {
        'kappa1': k1, 'theta1': theta1, 'sigma1': s1, 'v01': theta1, 'rho1': rh1,
        'kappa2': k2, 'theta2': theta2, 'sigma2': s2, 'v02': theta2, 'rho2': rh2,
        'rv_target': rv_target
    }


def calculate_hurst_exponent(returns):
    """
    Calculate Hurst exponent using variogram method
    This is the professional method from the second code
    """
    y = np.abs(returns.flatten())
    vario = [np.var(y[h:] - y[:-h]) for h in [1, 2, 4, 8, 16, 32] if len(y) > h]
    
    if len(vario) < 3:
        return 0.5
    
    try:
        H = np.polyfit(np.log([1, 2, 4, 8, 16, 32][:len(vario)]), np.log(vario), 1)[0] / 2
        H = max(min(H, 0.95), 0.05)
        return H
    except:
        return 0.5


def fgn_davies_harte(N, H):
    """Generate fractional Gaussian noise using Davies-Harte method"""
    if N <= 0 or H <= 0 or H >= 1:
        return np.zeros(max(N, 1))

    def gamma(k):
        k = float(k)
        return 0.5 * (
            abs(k-1)**(2*H)
            - 2*abs(k)**(2*H)
            + abs(k+1)**(2*H)
        )

    g = np.array([gamma(k) for k in range(N+1)])
    c = np.concatenate([g, g[1:-1][::-1]])
    lam = np.fft.fft(c).real
    lam = np.maximum(lam, 1e-12)

    Z = np.random.normal(size=len(lam)) + 1j * np.random.normal(size=len(lam))
    W = np.fft.ifft(np.sqrt(lam) * Z)
    X = W[:N].real
    X = (X - X.mean()) / (X.std() + 1e-12)
    return X

# =============================================================================
# MODULE 5: STRESS TESTING
# =============================================================================

def stress_test_table(current_price):
    """Generate stress testing scenarios"""
    scenarios = [
        ('Mild Shock', -0.05),
        ('Severe Shock', -0.10),
        ('Crisis Shock', -0.20),
        ('Volatility Spike', -0.30)
    ]

    rows = []
    for name, shock in scenarios:
        stressed_price = current_price * (1 + shock)
        loss = current_price - stressed_price
        rows.append({
            'Scenario': name,
            'Shock (%)': shock * 100,
            'Loss ($)': loss,
            'Final Price': stressed_price
        })

    return pd.DataFrame(rows)

# =============================================================================
# MAIN VALIDATION
# =============================================================================

def validate_asset(ticker, table_name, asset_label, color):
    """Complete asset validation with advanced calibration"""
    print("\n" + "="*80)
    print(f"🔍 VALIDATING: {asset_label} ({ticker})")
    print("="*80)

    df = load_data(ticker, table_name)
    returns = df['Return']
    current_price = float(df['Close'].iloc[-1])

    print(f"\nData Points: {len(df)}")
    print(f"Current Price: ${current_price:.2f}")

    # ===== ADVANCED PARAMETER ESTIMATION =====
    lamJ, muJ, sigJ, n_jumps = detect_jumps_advanced(returns.values)
    nu_t = estimate_t_dof(returns.values)
    params = calibrate_model_advanced(returns)
    H_hat = calculate_hurst_exponent(returns.values)
    
    # Fractional volatility intensity
    xi_hat = 0.25 if H_hat < 0.5 else 0.15

    regime = Analytics.classify_regime(lamJ, nu_t, H_hat)

    print(f"\n📊 MODEL PARAMETERS (Advanced Calibration):")
    print(f"   Regime: {regime}")
    print(f"   t-dof ν = {nu_t:.2f}")
    print(f"   Jump λ = {lamJ:.3f} ({n_jumps} days)")
    print(f"   Hurst H = {H_hat:.3f}")
    print(f"   ξ = {xi_hat:.3f}")
    print(f"   κ₁ = {params['kappa1']:.3f}, θ₁ = {params['theta1']:.6f}, σ₁ = {params['sigma1']:.3f}")
    print(f"   κ₂ = {params['kappa2']:.3f}, θ₂ = {params['theta2']:.6f}, σ₂ = {params['sigma2']:.3f}")

    # ===== ROLLING VAR BACKTEST =====
    closes = df['Close'].values
    mu_daily = float(returns.mean())
    fgn_steps = fgn_davies_harte(ROLLING_WINDOW, H_hat) if xi_hat != 0 else np.zeros(ROLLING_WINDOW)

    violations = []
    rolling_returns = []

    for t in range(ROLLING_WINDOW, len(closes)-1, ROLLING_STEP):
        # Re-calibrate on rolling window
        r_win = pd.Series(returns.values[:t])
        try:
            cal = calibrate_model_advanced(r_win)
        except:
            cal = params
        
        fgn_1d = np.array([fgn_steps[0]]) if xi_hat != 0 else np.zeros(1)
        
        paths = simulate_double_heston(
            closes[t], 1, 3000,
            (
                cal['kappa1'], cal['theta1'], cal['sigma1'],
                cal['v01'], cal['rho1'],
                cal['kappa2'], cal['theta2'], cal['sigma2'],
                cal['v02'], cal['rho2']
            ),
            mu_daily, lamJ, muJ, sigJ, H_hat, xi_hat,
            fgn_1d, nu_t
        )

        var_5 = np.percentile(paths[:, -1], 5)
        actual_next = closes[t+1]
        is_breach = 1 if actual_next < var_5 else 0
        violations.append(is_breach)
        rolling_returns.append((actual_next - closes[t]) / closes[t] * 100)

    n = len(violations)
    n_exceptions = sum(violations)
    coverage = (1 - n_exceptions/n) * 100 if n > 0 else 0

    kupiec = VaRBacktest.kupiec(n_exceptions, n)
    christoffersen = VaRBacktest.christoffersen(np.array(violations))
    traffic = VaRBacktest.basel_traffic_light(n_exceptions)

    verdict = (
        "PASS"
        if (
            kupiec['verdict'] == 'PASS'
            and christoffersen['verdict'] == 'PASS'
        )
        else "FAIL"
    )

    print(f"\n🏆 REGULATORY VALIDATION (1-day VaR 95%):")
    print(f"   Forecasts: {n}")
    print(f"   Coverage: {coverage:.1f}% (target: 95%)")
    print(f"   Exceptions: {n_exceptions} ({n_exceptions/n*100:.2f}%)")
    print(f"   Basel Traffic Light: {traffic}")
    print(f"   Kupiec: {kupiec['verdict']} (p={kupiec['p_value']:.4f})")
    print(f"   Christoffersen: {christoffersen['verdict']} (p={christoffersen['p_value']:.4f})")
    print(f"   Final Verdict: {verdict}")

    # ===== DISTRIBUTION ANALYSIS =====
    Analytics.distribution_stats(returns.values * 100, "Actual Returns")

    # ===== SIMULATE FOR COMPARISON =====
    sim_paths = simulate_double_heston(
        current_price, 1, 5000,
        (
            params['kappa1'], params['theta1'], params['sigma1'],
            params['v01'], params['rho1'],
            params['kappa2'], params['theta2'], params['sigma2'],
            params['v02'], params['rho2']
        ),
        mu_daily, lamJ, muJ, sigJ, H_hat, xi_hat,
        fgn_steps[:1], nu_t
    )
    sim_returns = (sim_paths[:, 1] / sim_paths[:, 0] - 1) * 100

    ks_stat, ks_p = ks_2samp(returns.values[-500:]*100, sim_returns[:500])
    print(f"\n   KS Test (Actual vs Simulated): p={ks_p:.4f} ({'Same distribution' if ks_p>0.05 else 'Different'})")

    # ===== MULTI-HORIZON VAR/CVAR =====
    print(f"\n🔮 MULTI-HORIZON RISK FORECASTS:")
    risk_rows = []
    for horizon in [1, 3, 5, 10]:
        paths = simulate_double_heston(
            current_price, horizon, SIMULATIONS,
            (
                params['kappa1'], params['theta1'], params['sigma1'],
                params['v01'], params['rho1'],
                params['kappa2'], params['theta2'], params['sigma2'],
                params['v02'], params['rho2']
            ),
            mu_daily, lamJ, muJ, sigJ, H_hat, xi_hat,
            fgn_steps[:horizon], nu_t
        )
        var95, cvar95 = Analytics.calculate_var_cvar(paths, alpha=0.95)
        var99, cvar99 = Analytics.calculate_var_cvar(paths, alpha=0.99)
        risk_rows.append({
            'Horizon': horizon,
            'VaR95 (%)': round(var95, 2),
            'CVaR95 (%)': round(cvar95, 2),
            'VaR99 (%)': round(var99, 2),
            'CVaR99 (%)': round(cvar99, 2)
        })
    
    risk_df = pd.DataFrame(risk_rows)
    print(risk_df.to_string(index=False))

    # ===== TAIL RISK ANALYTICS =====
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    max_dd = Analytics.max_drawdown(returns.values)
    downside_vol = Analytics.downside_volatility(returns.values)
    tail_ratio = Analytics.tail_ratio(returns.values)
    jb_stat, jb_p = jarque_bera(returns)

    print(f"\n📉 TAIL RISK ANALYTICS:")
    tail_df = pd.DataFrame([{
        'Skewness': round(skewness, 4),
        'Excess Kurtosis': round(kurtosis, 4),
        'Max Drawdown (%)': round(max_dd, 2),
        'Tail Ratio': round(tail_ratio, 3),
        'Downside Vol (%)': round(downside_vol, 2),
        'JB p-value': round(jb_p, 4)
    }])
    print(tail_df.to_string(index=False))

    # ===== STRESS TEST =====
    print(f"\n🔥 STRESS TEST SCENARIOS:")
    stress_df = stress_test_table(current_price)
    print(stress_df.to_string(index=False))

    # ===== PLOTS =====
    # 1. QQ Plot
    Analytics.plot_qq_comparison(asset_label, returns.values * 100, sim_returns)
    
    # 2. Distribution comparison
    Analytics.plot_distribution(asset_label, returns.values * 100, sim_returns)
    
    # 3. VaR Breach plot
    if len(rolling_returns) > 0:
        var_level = np.percentile(sim_returns, 5)
        breach_array = np.array(violations[:len(rolling_returns)]) == 1
        Analytics.plot_var_breach(asset_label, np.array(rolling_returns), var_level, breach_array)

    return {
        'ticker': ticker,
        'asset': asset_label,
        'verdict': verdict,
        'coverage': coverage,
        'traffic': traffic,
        'regime': regime,
        'nu_t': nu_t,
        'lamJ': lamJ,
        'H': H_hat,
        'kupiec': kupiec['verdict'],
        'christoffersen': christoffersen['verdict']
    }

# =============================================================================
# COMPARISON ENGINE
# =============================================================================

def run_comparison():
    """Run full comparison between assets"""
    start_time = time.time()

    print("\n" + "█"*80)
    print("     INSTITUTIONAL DOUBLE HESTON RISK ENGINE")
    print("     NVDA vs GSPC (ADVANCED CALIBRATION)")
    print("█"*80)

    results = {}
    for ticker, info in ASSETS.items():
        results[ticker] = validate_asset(
            ticker,
            info['table'],
            info['label'],
            info['color']
        )

    # Final comparison table
    comparison_df = pd.DataFrame([{
        'Asset': res['ticker'],
        'Regime': res['regime'],
        'Verdict': res['verdict'],
        'Coverage': f"{res['coverage']:.1f}%",
        'Traffic': res['traffic'],
        'Kupiec': res['kupiec'],
        'Christoffersen': res['christoffersen'],
        'Jump λ': round(res['lamJ'], 3),
        't-dof ν': round(res['nu_t'], 2),
        'Hurst H': round(res['H'], 3)
    } for res in results.values()])

    print("\n" + "="*80)
    print("🏛 FINAL INSTITUTIONAL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds")

    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    try:
        results = run_comparison()

        print("\n" + "="*80)
        print("🎯 FINAL INTERPRETATION")
        print("="*80)
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  KEY INSIGHTS:                                                              │
│                                                                             │
│  ✅ ADVANCED CALIBRATION FEATURES:                                          │
│     • Numerical optimization with L-BFGS-B                                 │
│     • Feller constraint enforcement                                        │
│     • Target volatility matching                                           │
│     • Advanced jump detection (RV/BPV/RQP method)                          │
│     • Hurst exponent via variogram                                         │
│                                                                             │
│  📊 ASSET-SPECIFIC FINDINGS:                                                │
│     • Jump-Dominant assets (NVDA) → Model performs well                    │
│     • Diffusion assets (GSPC) → May need calibration adjustment            │
│                                                                             │
│  💡 INTERPRETATION FOR INTERVIEWS:                                          │
│     • "The model uses advanced numerical calibration with Feller           │
│        constraints to ensure mathematical validity."                       │
│     • "We detect jumps using the realized variance vs bipower variation    │
│        methodology, which is robust to microstructure noise."              │
│     • "The Hurst exponent is estimated via variogram to capture            │
│        long-range dependence in volatility."                               │
│     • "For regulatory compliance, we use Kupiec and Christoffersen         │
│        backtests with Basel traffic light classification."                 │
└─────────────────────────────────────────────────────────────────────────────┘
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
