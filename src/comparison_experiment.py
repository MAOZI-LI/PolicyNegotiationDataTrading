"""Revised Fig. 6 comparison experiment.

This script builds a reproducible, single-run comparison among the proposed
Stackelberg mechanism and three adapted baselines. It intentionally writes a new
figure instead of replacing the paper's existing compare.eps.
"""

import argparse
import contextlib
import copy
import io
import math
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from data_trading_game import create_market, compute_best_strategy_and_utilities
import robustness_utils as robustness_baselines

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 9,
    'font.weight': 'normal',
    'axes.labelsize': 10,
    'axes.labelweight': 'normal',
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
})


FIG_DIR = 'figures/comparison_revised'
RESULT_DIR = 'results/comparison_revised'
CSV_PATH = os.path.join(RESULT_DIR, 'comparison_single_run.csv')
AUDIT_PATH = os.path.join(RESULT_DIR, 'comparison_audit.txt')
PDF_PATH = os.path.join(FIG_DIR, 'compare_revised.pdf')

MECHANISMS = ('Proposed', 'SRC', 'BGM', 'APM')
DISPLAY_LABELS = {
    'Proposed': 'Our Model',
    'SRC': 'SRC',
    'BGM': 'BGM',
    'APM': 'APM',
}
COLORS = {
    'Proposed': '#1f77b4',
    'SRC': '#d62728',
    'BGM': '#2ca02c',
    'APM': '#9467bd',
}
MARKERS = {
    'Proposed': 'o',
    'SRC': 's',
    'BGM': '^',
    'APM': 'D',
}
_BASE_MARKET = None


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_market_quiet():
    global _BASE_MARKET
    if _BASE_MARKET is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _BASE_MARKET = create_market(decay_type='linear', delivery_time_type='log')
    return copy.deepcopy(_BASE_MARKET)


def apply_varied_parameter(market, name, value):
    if name == 'lambda':
        market.lamda = float(value)
    elif name == 't1_t0':
        market.t1_t0 = float(value)
        market.t1_abs = market.t1_t0 * market.t0_abs
    else:
        raise ValueError(f'unknown varied parameter: {name}')
    market.clean_computation_cache()


def attr_psr(attr):
    return (attr['x'] + attr['z']) / attr['L_size'] if attr['L_size'] > 0 else 0.0


def avg_psr(market):
    values = [attr_psr(attr) for attr in market.attributes.values() if attr['L_size'] > 0]
    return float(np.mean(values)) if values else 0.0


def vector_str(values):
    return ';'.join(f'{float(v):.8g}' for v in values)


def current_vectors(market):
    p2s, zs = [], []
    for attr in market.attributes.values():
        p2s.append(attr['p2'])
        zs.append(attr['z'])
    return p2s, zs


def row_from_market(mechanism, varied_parameter, varied_value, market, feasible,
                    baseline_rule, extra=None):
    p2s, zs = current_vectors(market)
    if feasible:
        usc = market.buyer_utility()
        utsp = market.broker_utility()
        udp = market.seller_utility()
        quality = market.data_quality()
        delivery_time = market.delivery_time()
        time_ratio = market.time_ratio()
        risk_cost = market.seller_risk_cost()
        p_s = market.p1
        avg_p_t = float(np.mean(p2s)) if p2s else 0.0
        psr = avg_psr(market)
    else:
        usc = utsp = udp = quality = delivery_time = time_ratio = risk_cost = np.nan
        p_s = avg_p_t = psr = np.nan
    row = {
        'mechanism': mechanism,
        'varied_parameter': varied_parameter,
        'varied_value': varied_value,
        'feasible': bool(feasible),
        'USC': usc,
        'UTSP': utsp,
        'UDP': udp,
        'delivery_time': delivery_time,
        'time_ratio': time_ratio,
        'quality_q': quality,
        'risk_cost': risk_cost,
        'pS': p_s,
        'avg_pT': avg_p_t,
        'avg_psr': psr,
        'pT': vector_str(p2s) if p2s else '',
        'z': vector_str(zs) if zs else '',
        'baseline_rule': baseline_rule,
        'decay_type': market.decay_type,
        'delivery_time_type': market.delivery_time_type,
    }
    if extra:
        row.update(extra)
    return row


def set_policy_by_theta(market, theta, selected=None, low_theta=None):
    selected = set(market.attributes.keys()) if selected is None else set(selected)
    for name, attr in market.attributes.items():
        local_theta = theta if name in selected else (0.0 if low_theta is None else low_theta)
        local_theta = min(max(float(local_theta), 0.0), 1.0)
        x = attr['L_size'] - attr['L_join_B_size']
        y = 0
        z = local_theta * attr['L_join_B_size']
        if attr['type'] == 'discrete':
            z = int(round(z))
        market.set_policy(name, x, y, min(z, attr['L_join_B_size']))


def risk_compensation_prices(market, margin=0.2, selected=None, auction_floor=None):
    selected = set(market.attributes.keys()) if selected is None else set(selected)
    auction_floor = auction_floor or {}
    p2s = {}
    for name, attr in market.attributes.items():
        psr = attr_psr(attr)
        if name not in selected or psr <= 0 or attr['B_size'] <= 0:
            p2 = 0.0
        else:
            risk_i = market.lamda * market.data_size * attr['rho'] * ((attr['y'] + attr['z']) / attr['B_size']) ** 2
            p2 = (1.0 + margin) * risk_i / max(market.data_size * psr, 1e-12)
        p2 = max(p2, auction_floor.get(name, 0.0))
        market.set_p2(name, p2)
        p2s[name] = p2
    return p2s


def set_consumer_price_min_broker(market, eps=1e-9):
    q = market.data_quality()
    if q <= 0:
        return False
    min_payment = market.broker_cost() / q
    market.p1 = max(0.0, min_payment + eps)
    return True


def is_feasible(market):
    return (
        np.isfinite(market.buyer_utility()) and
        market.buyer_utility() >= -1e-8 and
        market.broker_utility() >= -1e-8 and
        market.seller_utility() >= -1e-8 and
        market.data_quality() > 0
    )


def better_candidate(row, best, metric='USC'):
    if not row['feasible']:
        return False
    if best is None:
        return True
    return row[metric] > best[metric] + 1e-9


def proposed_row(varied_parameter, varied_value):
    market = make_market_quiet()
    apply_varied_parameter(market, varied_parameter, varied_value)
    attrs = list(market.attributes.keys())
    ret = compute_best_strategy_and_utilities(market, attrs)
    feasible = ret['buyer_utility'] >= 0 and ret['broker_utility'] >= 0 and ret['seller_utility'] >= 0
    return row_from_market(
        'Proposed', varied_parameter, varied_value, market, feasible,
        'Full three-stage Stackelberg backward induction.',
        {'solver': 'analytical_main_model'},
    )


def reference_risk_cap_for_market(varied_parameter, varied_value):
    ref = make_market_quiet()
    apply_varied_parameter(ref, varied_parameter, varied_value)
    compute_best_strategy_and_utilities(ref, list(ref.attributes.keys()))
    return max(
        robustness_baselines.avg_risk_value(ref) * robustness_baselines.FAIR_RISK_CAP_MULTIPLIER,
        1e-12,
    )


def src_row(varied_parameter, varied_value, theta_grid=None, margin_grid=None):
    market = make_market_quiet()
    apply_varied_parameter(market, varied_parameter, varied_value)
    risk_cap = reference_risk_cap_for_market(varied_parameter, varied_value)
    try:
        robustness_baselines._run_src(market, risk_cap)
        feasible = is_feasible(market)
    except Exception:
        feasible = False
    return row_from_market(
        'SRC', varied_parameter, varied_value, market, feasible,
        'Static risk compensation baseline aligned with the statistical robustness experiment: fixed policy level, common risk cap, risk compensation, and positive TSP surplus.',
        {'risk_cap': risk_cap},
    )


def bgm_row(varied_parameter, varied_value, theta_grid=None, margin_grid=None):
    market = make_market_quiet()
    apply_varied_parameter(market, varied_parameter, varied_value)
    risk_cap = reference_risk_cap_for_market(varied_parameter, varied_value)
    try:
        robustness_baselines._run_bgm(market, risk_cap)
        feasible = is_feasible(market)
    except Exception:
        feasible = False
    return row_from_market(
        'BGM', varied_parameter, varied_value, market, feasible,
        'Bargaining-style baseline aligned with the statistical robustness experiment: risk-weighted allocation, common risk cap, risk compensation, and positive TSP surplus.',
        {'risk_cap': risk_cap},
    )


def marginal_quality_scores(market):
    scores = []
    base_theta = 0.10
    for name, attr in market.attributes.items():
        trial = make_market_quiet()
        trial.lamda = market.lamda
        trial.t1_t0 = market.t1_t0
        trial.t1_abs = trial.t1_t0 * trial.t0_abs
        set_policy_by_theta(trial, base_theta)
        risk_compensation_prices(trial, margin=0.0)
        base_q = trial.data_quality()
        set_policy_by_theta(trial, base_theta, selected=set(trial.attributes.keys()) - {name}, low_theta=1.0)
        # The call above makes all but name low in selected semantics; set directly for clarity below.
        set_policy_by_theta(trial, 1.0, selected={name}, low_theta=base_theta)
        risk_compensation_prices(trial, margin=0.0)
        full_q = trial.data_quality()
        scores.append((name, max(full_q - base_q, 0.0)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def apm_row(varied_parameter, varied_value, theta_grid=None, margin_grid=None):
    market = make_market_quiet()
    apply_varied_parameter(market, varied_parameter, varied_value)
    risk_cap = reference_risk_cap_for_market(varied_parameter, varied_value)
    try:
        robustness_baselines._run_apm(market, risk_cap)
        feasible = is_feasible(market)
    except Exception:
        feasible = False
    return row_from_market(
        'APM', varied_parameter, varied_value, market, feasible,
        'Auction-style baseline aligned with the statistical robustness experiment: marginal-match ranking, common risk cap, risk compensation, and second-price-style floor.',
        {'risk_cap': risk_cap},
    )


def infeasible_row(mechanism, varied_parameter, varied_value, rule):
    return {
        'mechanism': mechanism,
        'varied_parameter': varied_parameter,
        'varied_value': varied_value,
        'feasible': False,
        'USC': np.nan,
        'UTSP': np.nan,
        'UDP': np.nan,
        'delivery_time': np.nan,
        'time_ratio': np.nan,
        'quality_q': np.nan,
        'risk_cost': np.nan,
        'pS': np.nan,
        'avg_pT': np.nan,
        'avg_psr': np.nan,
        'pT': '',
        'z': '',
        'baseline_rule': rule,
        'decay_type': 'linear',
        'delivery_time_type': 'log',
    }


def run_experiment(fast=False):
    ensure_dirs()
    n_points = 11 if fast else 41
    lambda_values = np.linspace(0.4, 0.8, n_points)
    t_values = np.linspace(0.3, 0.5, n_points)
    theta_grid = np.linspace(0.25, 1.0, 7 if fast else 11)
    margin_grid = (0.05, 0.15, 0.30) if fast else (0.05, 0.10, 0.20, 0.30, 0.50)

    rows = []
    scans = [('lambda', lambda_values), ('t1_t0', t_values)]
    for varied_parameter, values in scans:
        for value in values:
            rows.append(proposed_row(varied_parameter, value))
            rows.append(src_row(varied_parameter, value))
            rows.append(bgm_row(varied_parameter, value))
            rows.append(apm_row(varied_parameter, value))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    plot_comparison(df)
    write_audit(df, fast, n_points, theta_grid, margin_grid)
    return df


def plot_comparison(df):
    style = robustness_baselines._stakeholder_fig_style('stakeholder_metrics')
    with robustness_baselines._paper_font_context(style):
        fig, axs = plt.subplots(2, 2, figsize=style['figsize'])
        panels = [
            ('lambda', 0.50, 'USC', 'Service Consumer Utility', r'$\lambda=0.5$', '(a)'),
            ('lambda', 0.50, 'UTSP', 'TSP Utility', r'$\lambda=0.5$', '(b)'),
            ('lambda', 0.50, 'UDP', 'Data Provider Utility', r'$\lambda=0.5$', '(c)'),
            ('t1_t0', 0.40, 'delivery_time', 'Delivery Time', r'$t_1/t_0=0.4$', '(d)'),
        ]
        x = np.arange(len(MECHANISMS))
        for ax, (param, target, metric, ylabel, setting_label, tag) in zip(axs.ravel(), panels):
            sub = df[df['varied_parameter'] == param].copy()
            nearest = sub.iloc[(sub['varied_value'] - target).abs().argsort()]['varied_value'].iloc[0]
            cur = sub[sub['varied_value'] == nearest].set_index('mechanism').reindex(MECHANISMS)
            values = cur[metric].where(cur['feasible']).to_numpy(dtype=float)
            bars = ax.bar(
                x, values, width=style['bar_width'],
                color=[COLORS[m] for m in MECHANISMS],
                alpha=0.85,
                edgecolor='white',
                linewidth=0.7,
            )
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                label = f'{value:.2f}' if abs(value) >= 10 else f'{value:.3f}'
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=style['bar_label_fs'],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [DISPLAY_LABELS[m] for m in MECHANISMS],
                fontsize=style['tick_fs'],
            )
            ax.set_ylabel(ylabel, fontsize=style['label_fs'])
            ax.set_title(f'{tag} {ylabel} ({setting_label})', fontsize=style['title_fs'], pad=6)
            ax.grid(True, axis='y', linestyle='--', alpha=0.35)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='both', labelsize=style['tick_fs'])
            finite = values[np.isfinite(values)]
            if len(finite):
                ymin = 0.0
                ymax = float(np.max(finite))
                if metric == 'delivery_time':
                    ymin = max(0.0, float(np.min(finite)) - 0.06)
                ax.set_ylim(ymin, ymax + 0.16 * max(ymax - ymin, 1e-9))
        fig.subplots_adjust(
            left=style['left'], right=style['right'],
            top=style['top'], bottom=style['bottom'],
            wspace=style['wspace'], hspace=style['hspace'],
        )
        fig.savefig(PDF_PATH, bbox_inches='tight', pad_inches=0.04)
        plt.close(fig)


def write_audit(df, fast, n_points, theta_grid, margin_grid):
    lines = [
        'Revised Fig. 6 comparison experiment audit',
        '===========================================',
        f'Fast mode: {fast}',
        f'Grid points per parameter: {n_points}',
        'Quality decay: linear benchmark.',
        'Delivery-time function: logarithmic benchmark.',
        'Dataset and market generation: create_market() with fixed seed 1234.',
        'Varied parameters: lambda in [0.4,0.8], t1/t0 in [0.3,0.5].',
        'Baseline rules: aligned with robustness_utils.py statistical robustness baselines.',
        f'Common risk cap multiplier: {robustness_baselines.FAIR_RISK_CAP_MULTIPLIER}',
        '',
        'Mechanism definitions:',
        '- Proposed: current three-stage Stackelberg backward induction.',
        f'- SRC: static risk compensation with fixed theta={robustness_baselines.SRC_FIXED_THETA}, common risk cap, risk margin={robustness_baselines.BASELINE_RISK_MARGIN}, and positive TSP surplus.',
        f'- BGM: bargaining-style risk-weighted allocation with theta={robustness_baselines.BGM_FIXED_THETA}, common risk cap, risk margin={robustness_baselines.BASELINE_RISK_MARGIN}, and positive TSP surplus.',
        f'- APM: auction-style allocation ranked by marginal match contribution, top fraction={robustness_baselines.APM_SELECTED_FRACTION}, theta={robustness_baselines.APM_FIXED_THETA}, low theta={robustness_baselines.APM_LOW_THETA}, and second-price-style floor.',
        '',
        'Feasibility counts:',
    ]
    counts = df.groupby('mechanism')['feasible'].agg(['sum', 'count'])
    for mechanism, row in counts.iterrows():
        lines.append(f'- {mechanism}: {int(row["sum"])}/{int(row["count"])} feasible rows')
    lines.extend([
        '',
        'Important caveat:',
        'These baselines are adapted to the unified policy-negotiation task. They are not full reimplementations of the original SRC/BGM/APM papers.',
        '',
        f'CSV: {CSV_PATH}',
        f'PDF: {PDF_PATH}',
    ])
    with open(AUDIT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Use fewer grid points for quick validation.')
    args = parser.parse_args()
    df = run_experiment(fast=args.fast)
    print(f'[ok] rows: {len(df)}')
    print(f'[ok] CSV: {CSV_PATH}')
    print(f'[ok] PDF: {PDF_PATH}')
    print(f'[ok] audit: {AUDIT_PATH}')


if __name__ == '__main__':
    main()
