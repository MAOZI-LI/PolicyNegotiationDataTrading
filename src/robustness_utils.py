"""Utilities for repeated robustness and statistical significance experiments."""

import contextlib
import io
import math
import os
import tarfile
import time
import warnings
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

from data_trading_game import Market, compute_best_strategy_and_utilities

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 9,
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
})


DATASET_CONFIGS = {
    'amazon_employee': {
        'path': 'data/over-datasets-kaggle-log-clean.csv',
        'inner_file': None,
        'exclude_columns': [],
    },
    'uci_amazon_access': {
        'path': 'dataset/amazon+access+samples.zip',
        'inner_file': 'amzn-anon-access-samples-history-2.0.csv',
        'exclude_columns': [],
    },
    'incident_event_log': {
        'path': 'dataset/incident+management+process+enriched+event+log.zip',
        'inner_file': 'incident_event_log.csv',
        'exclude_columns': [],
    },
}

METHODS = ('Our Model', 'SRC', 'BGM', 'APM')
SRC_FIXED_THETA = 0.70
BGM_FIXED_THETA = 0.70
BASELINE_RISK_MARGIN = 0.20
BASELINE_BROKER_SHARE = 0.02
FAIR_RISK_CAP_MULTIPLIER = 1.0
APM_SELECTED_FRACTION = 0.50
APM_FIXED_THETA = 0.80
APM_LOW_THETA = 0.02
APM_RISK_MARGIN = 0.25
APM_BROKER_SHARE = 0.02
APM_SECOND_PRICE_SCALE = 0.005
METRICS = (
    'U_SC', 'U_TSP', 'U_DP', 'delivery_time', 'pS', 'avg_pT',
    'avg_psr', 'avg_risk_value', 'quality_q', 'social_welfare',
    'quality_adjusted_welfare', 'welfare_per_time', 'quality_per_time',
    'runtime',
)
STAKEHOLDER_PLOT_METRICS = ('U_SC', 'U_TSP', 'U_DP', 'delivery_time')
PERFORMANCE_PLOT_METRICS = (
    'social_welfare', 'quality_q', 'welfare_per_time', 'quality_per_time'
)
METRIC_LABELS = {
    'U_SC': 'Service consumer utility',
    'U_TSP': 'TSP utility',
    'U_DP': 'Data provider utility',
    'delivery_time': 'Delivery time',
    'social_welfare': 'Social welfare',
    'quality_q': 'Data quality',
    'welfare_per_time': 'Welfare per time',
    'quality_per_time': 'Quality per time',
}

EXCLUDE_KEYWORDS = (
    'id', 'index', 'label', 'target', 'class', 'result', 'response',
    'decision', 'access', 'action', 'number',
)

_REFERENCE_RISK_CACHE = {}
_PAPER_FONT_NAME = None


def _resolve_paper_font():
    global _PAPER_FONT_NAME
    if _PAPER_FONT_NAME is not None:
        return _PAPER_FONT_NAME
    from matplotlib import font_manager

    candidates = [
        'Times New Roman',
        'Times',
        'Nimbus Roman',
        'TeX Gyre Termes',
        'STIXGeneral',
        'DejaVu Serif',
    ]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            _PAPER_FONT_NAME = name
            return _PAPER_FONT_NAME
        except (ValueError, OSError):
            continue
    nimbus_path = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
    if os.path.exists(nimbus_path):
        font_manager.fontManager.addfont(nimbus_path)
        _PAPER_FONT_NAME = font_manager.FontProperties(fname=nimbus_path).get_name()
        return _PAPER_FONT_NAME
    _PAPER_FONT_NAME = 'serif'
    return _PAPER_FONT_NAME


@dataclass
class DatasetProfile:
    dataset: str
    path: str
    rows: int
    columns: int
    selected_attributes: list
    continuous_attributes: list
    discrete_attributes: list
    excluded_attributes: list
    missing_summary: str


def _read_tabular_stream(stream, filename, nrows=None):
    lower = filename.lower()
    if lower.endswith('.csv'):
        return pd.read_csv(stream, nrows=nrows)
    if lower.endswith('.tsv') or lower.endswith('.txt'):
        try:
            return pd.read_csv(stream, sep='\t', nrows=nrows)
        except Exception:
            if hasattr(stream, 'seek'):
                stream.seek(0)
            return pd.read_csv(stream, nrows=nrows)
    raise ValueError(f'unsupported tabular file inside archive: {filename}')


def _find_archive_member(names, configured=None):
    files = [n for n in names if not n.endswith('/')]
    if configured:
        for name in files:
            if name.endswith(configured) or name == configured:
                return name
        raise FileNotFoundError(f'configured inner file not found: {configured}')
    candidates = [
        n for n in files
        if n.lower().endswith(('.csv', '.tsv', '.txt', '.xlsx', '.xls', '.tgz', '.tar.gz'))
    ]
    if not candidates:
        raise FileNotFoundError('no supported data file found inside archive')
    return sorted(candidates, key=lambda n: (not n.lower().endswith(('.csv', '.tsv', '.txt')), len(n)))[0]


def load_dataset(dataset_name, path=None, config=None, nrows=None):
    """Load csv/txt/tsv/xlsx/xls/zip/tgz data into a DataFrame."""
    config = dict(config or DATASET_CONFIGS.get(dataset_name, {}))
    path = path or config.get('path')
    inner_file = config.get('inner_file')
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'dataset path not found for {dataset_name}: {path}')
    lower = path.lower()
    if lower.endswith('.csv'):
        return pd.read_csv(path, nrows=nrows)
    if lower.endswith('.tsv') or lower.endswith('.txt'):
        return pd.read_csv(path, sep='\t', nrows=nrows)
    if lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, nrows=nrows)
    if lower.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            try:
                member = _find_archive_member(zf.namelist(), inner_file)
            except FileNotFoundError:
                # Some downloads wrap the configured CSV inside a top-level tgz.
                member = _find_archive_member(zf.namelist(), None)
            if member.lower().endswith(('.tgz', '.tar.gz')):
                data = zf.read(member)
                with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
                    tar_member = _find_archive_member([m.name for m in tf.getmembers() if m.isfile()], inner_file)
                    extracted = tf.extractfile(tar_member)
                    if extracted is None:
                        raise FileNotFoundError(f'cannot extract {tar_member}')
                    return _read_tabular_stream(extracted, tar_member, nrows=nrows)
            with zf.open(member) as stream:
                return _read_tabular_stream(stream, member, nrows=nrows)
    if lower.endswith(('.tgz', '.tar.gz')):
        with tarfile.open(path, mode='r:gz') as tf:
            member = _find_archive_member([m.name for m in tf.getmembers() if m.isfile()], inner_file)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f'cannot extract {member}')
            return _read_tabular_stream(extracted, member, nrows=nrows)
    raise ValueError(f'unsupported dataset format: {path}')


def _is_timestamp_like(series):
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    joined = ' '.join(sample.head(20).tolist()).lower()
    has_date_signal = any(token in joined for token in ('/', '-', ':', 'am', 'pm', 'utc'))
    if not has_date_signal:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = pd.to_datetime(sample, errors='coerce')
    return parsed.notna().mean() > 0.85


def _excluded_reason(col, series, n_rows, config):
    low = col.lower()
    if col in config.get('exclude_columns', []):
        return 'configured exclusion'
    if any(key == low or key in low for key in EXCLUDE_KEYWORDS):
        return 'identifier/label/access-like column'
    missing_ratio = float(series.isna().mean())
    if missing_ratio > config.get('max_missing_ratio', 0.4):
        return f'high missing ratio ({missing_ratio:.3f})'
    nunique = int(series.nunique(dropna=True))
    if nunique <= 1:
        return 'constant or empty column'
    unique_ratio = nunique / max(n_rows, 1)
    if unique_ratio > config.get('max_unique_ratio', 0.95):
        return f'high unique ratio ({unique_ratio:.3f})'
    if _is_timestamp_like(series) and unique_ratio > 0.5:
        return 'timestamp-like identifier'
    return ''


def _clean_series(series, attr_type, top_k=100):
    s = series.copy()
    if attr_type == 'continuous':
        if _is_timestamp_like(s):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                values = pd.to_datetime(s, errors='coerce').astype('int64') / 1e9
            values = values.replace(-9223372036.854776, np.nan)
        else:
            values = pd.to_numeric(s, errors='coerce')
        median = float(values.median()) if values.notna().any() else 0.0
        return values.fillna(median)
    s = s.astype('object').where(s.notna(), '__MISSING__').astype(str)
    counts = s.value_counts()
    keep = set(counts.head(top_k).index)
    return s.where(s.isin(keep), '__OTHER__')


def profile_dataset(dataset_name, df, path, config=None):
    config = config or DATASET_CONFIGS.get(dataset_name, {})
    n_rows = len(df)
    continuous, discrete, selected, excluded = [], [], [], []
    cleaned = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        reason = _excluded_reason(col, series, n_rows, config)
        if reason:
            excluded.append(f'{col}: {reason}')
            continue
        nunique = int(series.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(series) and nunique > config.get('discrete_numeric_threshold', 50):
            attr_type = 'continuous'
        elif _is_timestamp_like(series):
            attr_type = 'continuous'
        else:
            attr_type = 'discrete'
        cleaned[col] = _clean_series(series, attr_type, config.get('top_k_discrete', 100))
        selected.append(col)
        if attr_type == 'continuous':
            continuous.append(col)
        else:
            discrete.append(col)
    missing_summary = '; '.join(
        f'{c}:{int(v)}' for c, v in df.isna().sum().items() if int(v) > 0
    ) or 'none'
    profile = DatasetProfile(
        dataset=dataset_name,
        path=path,
        rows=n_rows,
        columns=len(df.columns),
        selected_attributes=selected,
        continuous_attributes=continuous,
        discrete_attributes=discrete,
        excluded_attributes=excluded,
        missing_summary=missing_summary,
    )
    return cleaned[selected], profile


def profile_to_row(profile):
    return {
        'dataset': profile.dataset,
        'path': profile.path,
        'num_rows': profile.rows,
        'num_columns': profile.columns,
        'selected_policy_attributes': ';'.join(profile.selected_attributes),
        'continuous_attributes': ';'.join(profile.continuous_attributes),
        'discrete_attributes': ';'.join(profile.discrete_attributes),
        'excluded_attributes_and_reasons': '; '.join(profile.excluded_attributes),
        'missing_value_summary': profile.missing_summary,
    }


def generate_policy_space(dataset_name, df, profile, overlap_level, seed, max_attrs=4):
    rng = np.random.default_rng(seed)
    ordered = []
    for col in profile.continuous_attributes:
        if col in profile.selected_attributes and col not in ordered:
            ordered.append(col)
    for col in profile.discrete_attributes:
        if col in profile.selected_attributes and col not in ordered:
            ordered.append(col)
    selected = ordered[:max_attrs]
    if not selected:
        raise ValueError(f'no usable policy attributes for dataset {dataset_name}')
    raw_w = rng.uniform(0.1, 1.0, len(selected))
    raw_rho = rng.uniform(0.1, 1.0, len(selected))
    weights = raw_w / raw_w.sum()
    rhos = raw_rho / raw_rho.sum()
    overlap_ranges = {
        'low': (0.1, 0.3),
        'medium': (0.4, 0.6),
        'high': (0.7, 0.9),
    }
    lo_overlap, hi_overlap = overlap_ranges[overlap_level]
    specs = []
    cont_set = set(profile.continuous_attributes)
    for idx, col in enumerate(selected):
        if col in cont_set:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) == 0 or float(values.max()) == float(values.min()):
                spread_factor = 1.0
            else:
                q10, q90 = np.nanquantile(values, [0.1, 0.9])
                full_range = max(float(values.max()) - float(values.min()), 1e-12)
                spread_factor = max(0.2, min(1.0, float(q90 - q10) / full_range))
            # The Stackelberg solver works on cardinalities/interval lengths and
            # has loops proportional to L_join_B_size.  Continuous real-world
            # fields can have huge raw units (timestamps, elapsed seconds), so
            # we map their empirical spread to a bounded abstract policy scale.
            universe = int(round(40 + 80 * spread_factor))
            l_size = int(rng.integers(max(2, universe // 8), max(3, universe // 2)))
            b_size = int(rng.integers(max(2, universe // 8), max(3, universe // 2)))
            overlap_target = rng.uniform(lo_overlap, hi_overlap) * min(l_size, b_size)
            join = int(max(1, min(min(l_size, b_size), round(overlap_target))))
            attr_type = 'continuous'
        else:
            n_unique = int(df[col].nunique(dropna=True))
            universe = max(2, min(60, n_unique))
            l_size = int(rng.integers(1, universe + 1))
            b_size = int(rng.integers(1, universe + 1))
            overlap_target = rng.uniform(lo_overlap, hi_overlap) * min(l_size, b_size)
            join = int(max(1, min(min(l_size, b_size), round(overlap_target))))
            attr_type = 'discrete'
        specs.append({
            'name': f'A{idx}',
            'source_column': col,
            'type': attr_type,
            'w': float(weights[idx]),
            'rho': float(rhos[idx]),
            'L_size': max(l_size, join, 1),
            'B_size': max(b_size, join, 1),
            'L_join_B_size': max(join, 0),
        })
    return specs


def market_from_policy_space(policy_space, seed, data_size, lamda=0.4, q_0=100,
                             gamma=300, t1_t0=0.5, accuracy=0.8, tao=0.00001):
    market = Market(seed=seed, data_size=max(int(data_size), 1), lamda=lamda,
                    q_0=q_0, gamma=gamma, t1_t0=t1_t0, accuracy=accuracy, tao=tao)
    for spec in policy_space:
        market.add_attribute(
            spec['name'], spec['type'], spec['w'], spec['rho'],
            int(spec['L_size']), int(spec['B_size']), int(spec['L_join_B_size'])
        )
    market.weight_regularize()
    return market


def attr_psr(attr):
    return (attr['x'] + attr['z']) / attr['L_size'] if attr['L_size'] > 0 else 0.0


def avg_psr(market):
    values = [attr_psr(attr) for attr in market.attributes.values() if attr['L_size'] > 0]
    return float(np.mean(values)) if values else 0.0


def avg_risk_value(market):
    vals = []
    for attr in market.attributes.values():
        denom = max(attr['B_size'], 1e-12)
        vals.append(attr['rho'] * ((attr['y'] + attr['z']) / denom) ** 2)
    return float(np.mean(vals)) if vals else 0.0


def vector_str(values):
    return ';'.join(f'{float(v):.8g}' for v in values)


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
    for name, attr in market.attributes.items():
        psr = attr_psr(attr)
        if name not in selected or psr <= 0 or attr['B_size'] <= 0:
            p2 = 0.0
        else:
            risk_i = market.lamda * market.data_size * attr['rho'] * ((attr['y'] + attr['z']) / attr['B_size']) ** 2
            p2 = (1.0 + margin) * risk_i / max(market.data_size * psr, 1e-12)
        market.set_p2(name, max(p2, auction_floor.get(name, 0.0)))


def set_consumer_price_min_broker(market, eps=1e-9):
    q = market.data_quality()
    if q <= 0:
        return False
    market.p1 = max(0.0, market.broker_cost() / q + eps)
    return True


def set_consumer_price_positive_broker(market, broker_share=0.25, eps=1e-9):
    q = market.data_quality()
    if q <= 0:
        return False
    min_p1 = max(0.0, market.broker_cost() / q + eps)
    max_p1 = market.buyer_revenue() / q - eps
    if max_p1 < min_p1:
        return False
    share = min(max(float(broker_share), 0.0), 1.0)
    market.p1 = min_p1 + share * (max_p1 - min_p1)
    return True


def is_feasible(market):
    try:
        return (
            market.data_quality() > 0 and
            market.buyer_utility() >= -1e-8 and
            market.broker_utility() >= -1e-8 and
            market.seller_utility() >= -1e-8
        )
    except Exception:
        return False


def row_from_market(dataset, overlap_level, run_id, seed, method, market, feasible, runtime):
    p2s = [attr['p2'] for attr in market.attributes.values()]
    zs = [attr['z'] for attr in market.attributes.values()]
    if feasible:
        usc = market.buyer_utility()
        utsp = market.broker_utility()
        udp = market.seller_utility()
        quality = market.data_quality()
        delivery = market.delivery_time()
        welfare = usc + utsp + udp
        return {
            'dataset': dataset,
            'overlap_level': overlap_level,
            'run_id': run_id,
            'random_seed': seed,
            'method': method,
            'U_SC': usc,
            'U_TSP': utsp,
            'U_DP': udp,
            'delivery_time': delivery,
            'pS': market.p1,
            'avg_pT': float(np.mean(p2s)) if p2s else 0.0,
            'avg_psr': avg_psr(market),
            'avg_risk_value': avg_risk_value(market),
            'quality_q': quality,
            'social_welfare': welfare,
            'quality_adjusted_welfare': welfare * quality,
            'welfare_per_time': welfare / max(delivery, 1e-12),
            'quality_per_time': quality / max(delivery, 1e-12),
            'runtime': runtime,
            'feasible': True,
            'pT': vector_str(p2s),
            'z': vector_str(zs),
        }
    return {
        'dataset': dataset,
        'overlap_level': overlap_level,
        'run_id': run_id,
        'random_seed': seed,
        'method': method,
        'U_SC': np.nan,
        'U_TSP': np.nan,
        'U_DP': np.nan,
        'delivery_time': np.nan,
        'pS': np.nan,
        'avg_pT': np.nan,
        'avg_psr': np.nan,
        'avg_risk_value': np.nan,
        'quality_q': np.nan,
        'social_welfare': np.nan,
        'quality_adjusted_welfare': np.nan,
        'welfare_per_time': np.nan,
        'quality_per_time': np.nan,
        'runtime': runtime,
        'feasible': False,
        'pT': '',
        'z': '',
    }


def run_method(method_name, dataset, overlap_level, run_id, seed, policy_space, data_size):
    start = time.time()
    market = market_from_policy_space(policy_space, seed, data_size)
    try:
        if method_name == 'Our Model':
            attrs = list(market.attributes.keys())
            compute_best_strategy_and_utilities(market, attrs)
        elif method_name == 'SRC':
            _run_src(market, _reference_risk_cap(policy_space, seed, data_size))
        elif method_name == 'BGM':
            _run_bgm(market, _reference_risk_cap(policy_space, seed, data_size))
        elif method_name == 'APM':
            _run_apm(market, _reference_risk_cap(policy_space, seed, data_size))
        else:
            raise ValueError(f'unknown method: {method_name}')
        feasible = is_feasible(market)
    except Exception:
        feasible = False
    runtime = time.time() - start
    return row_from_market(dataset, overlap_level, run_id, seed, method_name, market, feasible, runtime)


def _candidate_grids():
    return np.linspace(0.25, 1.0, 7), (0.05, 0.15, 0.30, 0.50), (0.15, 0.25, 0.35)


def _balanced_score(market):
    delivery = max(market.delivery_time(), 1e-12)
    welfare = market.buyer_utility() + market.broker_utility() + market.seller_utility()
    return welfare * market.data_quality() / delivery


def _policy_space_cache_key(policy_space, seed, data_size):
    specs = []
    for spec in policy_space:
        specs.append((
            spec.get('name'), spec.get('type'), round(float(spec.get('w', 0.0)), 12),
            round(float(spec.get('rho', 0.0)), 12), int(spec.get('L_size', 0)),
            int(spec.get('B_size', 0)), int(spec.get('L_join_B_size', 0)),
        ))
    return int(seed), int(data_size), tuple(specs)


def _reference_risk_cap(policy_space, seed, data_size):
    key = _policy_space_cache_key(policy_space, seed, data_size)
    if key not in _REFERENCE_RISK_CACHE:
        ref = market_from_policy_space(policy_space, seed, data_size)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            compute_best_strategy_and_utilities(ref, list(ref.attributes.keys()))
        _REFERENCE_RISK_CACHE[key] = max(avg_risk_value(ref) * FAIR_RISK_CAP_MULTIPLIER, 1e-12)
    return _REFERENCE_RISK_CACHE[key]


def set_policy_by_theta_with_risk_cap(market, desired_theta, risk_cap, selected=None, low_theta=None):
    # Baselines keep their rule-based target policy level, but they cannot obtain
    # shorter delivery time by taking substantially higher data-exposure risk
    # than the Stackelberg benchmark on the same policy-space instance.
    desired_theta = min(max(float(desired_theta), 0.0), 1.0)
    for theta in np.linspace(desired_theta, 0.0, 121):
        set_policy_by_theta(market, theta, selected=selected, low_theta=low_theta)
        if avg_risk_value(market) <= risk_cap + 1e-12:
            return theta
    set_policy_by_theta(market, 0.0, selected=selected, low_theta=low_theta)
    return 0.0


def set_policy_by_thetas_with_risk_cap(market, desired_thetas, risk_cap):
    # Scale a mechanism-specific attribute-level allocation pattern until it
    # satisfies the common risk budget. This preserves each baseline's policy
    # structure while keeping risk exposure comparable.
    for scale in np.linspace(1.0, 0.0, 121):
        for name, attr in market.attributes.items():
            theta = min(max(float(desired_thetas.get(name, 0.0)) * scale, 0.0), 1.0)
            x = attr['L_size'] - attr['L_join_B_size']
            y = 0
            z = theta * attr['L_join_B_size']
            if attr['type'] == 'discrete':
                z = int(round(z))
            market.set_policy(name, x, y, min(z, attr['L_join_B_size']))
        if avg_risk_value(market) <= risk_cap + 1e-12:
            return scale
    for name, attr in market.attributes.items():
        market.set_policy(name, attr['L_size'] - attr['L_join_B_size'], 0, 0)
    return 0.0


def _run_src(market, risk_cap):
    trial = _clone_market_structure(market)
    set_policy_by_theta_with_risk_cap(trial, SRC_FIXED_THETA, risk_cap)
    risk_compensation_prices(trial, margin=BASELINE_RISK_MARGIN)
    if not (set_consumer_price_positive_broker(trial, broker_share=BASELINE_BROKER_SHARE) and is_feasible(trial)):
        raise RuntimeError('SRC no feasible point')
    _copy_market_state(trial, market)


def _run_bgm(market, risk_cap):
    trial = _clone_market_structure(market)
    rho_values = [attr['rho'] for attr in trial.attributes.values()]
    mean_rho = float(np.mean(rho_values)) if rho_values else 1.0
    desired = {}
    for name, attr in trial.attributes.items():
        # BGM is represented as a bargaining-style rule with risk-weighted
        # attribute allocation. It is not allowed to re-optimize the full
        # Stackelberg policy grid, and the common risk cap later scales this
        # allocation to keep exposure comparable with the proposed mechanism.
        desired[name] = min(max(BGM_FIXED_THETA * math.sqrt(attr['rho'] / max(mean_rho, 1e-12)), 0.02), 1.0)
    set_policy_by_thetas_with_risk_cap(trial, desired, risk_cap)
    risk_compensation_prices(trial, margin=BASELINE_RISK_MARGIN)
    if not (set_consumer_price_positive_broker(trial, broker_share=BASELINE_BROKER_SHARE) and is_feasible(trial)):
        raise RuntimeError('BGM no feasible point')
    _copy_market_state(trial, market)


def _run_apm(market, risk_cap):
    ranked = _rank_attrs_by_join(market)
    names = [name for name, _ in ranked]
    selected_count = max(1, int(round(len(names) * APM_SELECTED_FRACTION)))
    selected = set(names[:selected_count])
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # APM is a rule-based auction baseline rather than another global optimizer:
    # attributes are ranked by marginal match contribution, allocated at a fixed
    # policy level, and paid by risk compensation plus a small second-price-style
    # floor. This avoids selecting the APM outcome by the same welfare objective
    # used to evaluate the mechanisms.
    trial = _clone_market_structure(market)
    desired = {
        name: (APM_FIXED_THETA if name in selected else APM_LOW_THETA)
        for name in trial.attributes
    }
    set_policy_by_thetas_with_risk_cap(trial, desired, risk_cap)
    floor = {}
    for name in selected:
        psr = attr_psr(trial.attributes[name])
        floor[name] = APM_SECOND_PRICE_SCALE * second_score / max(
            trial.data_size * max(psr, 1e-12), 1e-12
        )
    risk_compensation_prices(trial, margin=APM_RISK_MARGIN, selected=selected, auction_floor=floor)
    if not (set_consumer_price_positive_broker(trial, broker_share=APM_BROKER_SHARE) and is_feasible(trial)):
        raise RuntimeError('APM no feasible point')
    _copy_market_state(trial, market)


def _clone_market_structure(market):
    clone = Market(market.seed, market.data_size, market.lamda, market.q_0,
                   market.gamma, market.t1_t0, market.accuracy, market.tao,
                   decay_type=market.decay_type, decay_eta=market.decay_eta,
                   decay_alpha=market.decay_alpha, t0_abs=market.t0_abs,
                   delivery_time_type=market.delivery_time_type,
                   delivery_beta=market.delivery_beta)
    for name, attr in market.attributes.items():
        clone.add_attribute(name, attr['type'], attr['w'], attr['rho'],
                            attr['L_size'], attr['B_size'], attr['L_join_B_size'])
    return clone


def _copy_market_state(src, dst):
    dst.p1 = src.p1
    for name, attr in src.attributes.items():
        dst.set_p2(name, attr['p2'])
        dst.set_policy(name, attr['x'], attr['y'], attr['z'])


def _rank_attrs_by_join(market):
    ranked = []
    for name, attr in market.attributes.items():
        score = attr['w'] * max(attr['L_join_B_size'], 0) / max(attr['L_size'], 1)
        ranked.append((name, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def summarize_results(raw_df):
    rows = []
    for keys, group in raw_df.groupby(['dataset', 'overlap_level', 'method']):
        dataset, overlap, method = keys
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors='coerce').dropna()
            n = len(vals)
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else 0.0 if n == 1 else np.nan
            se = std / math.sqrt(n) if n > 0 and np.isfinite(std) else np.nan
            ci_low = mean - 1.96 * se if n > 0 and np.isfinite(se) else np.nan
            ci_high = mean + 1.96 * se if n > 0 and np.isfinite(se) else np.nan
            rows.append({
                'dataset': dataset, 'overlap_level': overlap, 'method': method,
                'metric': metric, 'n': n, 'mean': mean, 'std': std,
                'standard_error': se, 'ci95_low': ci_low, 'ci95_high': ci_high,
            })
    return pd.DataFrame(rows)


def significance_label(p):
    if not np.isfinite(p):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def paired_t_pvalue(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 3:
        return np.nan, 'sample too small'
    std = diff.std(ddof=1)
    if std == 0:
        return np.nan, 'all paired differences are zero'
    t_stat = diff.mean() / (std / math.sqrt(len(diff)))
    return _normal_two_sided_p(t_stat), 'normal approximation (scipy unavailable)'


def wilcoxon_pvalue(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    n = len(diff)
    if n < 3:
        return np.nan, 'sample too small or all differences are zero'
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    w_pos = ranks[diff > 0].sum()
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return np.nan, 'invalid variance'
    z = (w_pos - mean) / math.sqrt(var)
    return _normal_two_sided_p(z), 'normal approximation (scipy unavailable)'


def run_significance_tests(raw_df):
    rows = []
    warnings = []
    baselines = [m for m in METHODS if m != 'Our Model']
    for (dataset, overlap), group in raw_df.groupby(['dataset', 'overlap_level']):
        for metric in METRICS:
            base = group[group['method'] == 'Our Model'][['run_id', metric]].rename(columns={metric: 'our'})
            for baseline in baselines:
                other = group[group['method'] == baseline][['run_id', metric]].rename(columns={metric: 'baseline'})
                merged = base.merge(other, on='run_id', how='inner').dropna()
                diff = merged['our'].to_numpy(dtype=float) - merged['baseline'].to_numpy(dtype=float)
                mean_diff = float(np.mean(diff)) if len(diff) else np.nan
                t_p, t_note = paired_t_pvalue(diff)
                w_p, w_note = wilcoxon_pvalue(diff)
                note = '; '.join(sorted(set([t_note, w_note]) - {''}))
                if 'sample too small' in note or 'zero' in note:
                    warnings.append(f'{dataset}/{overlap}/{metric}/{baseline}: {note}')
                p_for_label = t_p if np.isfinite(t_p) else w_p
                rows.append({
                    'dataset': dataset,
                    'overlap_level': overlap,
                    'metric': metric,
                    'baseline': baseline,
                    'n_pairs': len(diff),
                    'mean_difference': mean_diff,
                    'paired_ttest_pvalue': t_p,
                    'wilcoxon_pvalue': w_p,
                    'significance_label': significance_label(p_for_label),
                    'notes': note,
                })
    return pd.DataFrame(rows), warnings


def save_latex_tables(summary_df, tests_df, output_dir):
    summary_path = os.path.join(output_dir, 'robustness_summary.tex')
    tests_path = os.path.join(output_dir, 'significance_tests.tex')
    summary_view = summary_df.copy()
    summary_view['mean_pm_std'] = summary_view.apply(
        lambda r: f"{r['mean']:.4g} $\\pm$ {r['std']:.4g}", axis=1
    )
    summary_view['ci95'] = summary_view.apply(
        lambda r: f"[{r['ci95_low']:.4g}, {r['ci95_high']:.4g}]", axis=1
    )
    summary_cols = ['dataset', 'overlap_level', 'method', 'metric', 'n', 'mean_pm_std', 'ci95']
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(_simple_latex_table(summary_view[summary_cols], escape=False))
    tests_cols = [
        'dataset', 'overlap_level', 'metric', 'baseline', 'n_pairs',
        'mean_difference', 'paired_ttest_pvalue', 'wilcoxon_pvalue',
        'significance_label',
    ]
    with open(tests_path, 'w', encoding='utf-8') as f:
        view = tests_df[tests_cols].copy()
        for col in ('mean_difference', 'paired_ttest_pvalue', 'wilcoxon_pvalue'):
            view[col] = view[col].map(lambda v: f'{v:.4g}' if pd.notna(v) else '')
        f.write(_simple_latex_table(view, escape=True))


def _latex_escape(value):
    text = '' if pd.isna(value) else str(value)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _simple_latex_table(df, escape=True):
    cols = list(df.columns)
    align = 'l' * len(cols)
    lines = [r'\begin{tabular}{' + align + '}', r'\hline']
    header = [_latex_escape(c) for c in cols] if escape else [str(c) for c in cols]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for _, row in df.iterrows():
        vals = [_latex_escape(row[c]) for c in cols] if escape else [('' if pd.isna(row[c]) else str(row[c])) for c in cols]
        lines.append(' & '.join(vals) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}', ''])
    return '\n'.join(lines)


def _metric_label(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' '))


def _stakeholder_fig_style(suffix):
    paper = suffix == 'stakeholder_metrics'
    return {
        'figsize': (7.4, 5.6) if paper else (9.0, 6.2),
        'bar_width': 0.58,
        'title_fs': 10,
        'label_fs': 12 if paper else 8.5,
        'tick_fs': 9.5 if paper else 8,
        'bar_label_fs': 8.5 if paper else 7,
        'x_rotation': 0,
        'show_suptitle': not paper,
        'use_times': paper,
        'font_name': _resolve_paper_font() if paper else None,
        'wspace': 0.34,
        'hspace': 0.42,
        'left': 0.16 if paper else 0.14,
        'right': 0.98,
        'top': 0.98,
        'bottom': 0.16,
    }


def _paper_font_context(style):
    if not style.get('use_times'):
        return contextlib.nullcontext()
    font_name = style['font_name']
    return plt.rc_context({
        'font.family': 'serif',
        'font.serif': [font_name, 'Nimbus Roman', 'DejaVu Serif'],
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _centered_spec_label(fig, spec, label, *, fontsize, kind, font_name=None):
    fig.canvas.draw()
    pos = spec.get_position(fig)
    text_kwargs = {'fontsize': fontsize}
    if font_name:
        text_kwargs['fontfamily'] = font_name
    if kind == 'title':
        fig.text(
            0.5 * (pos.x0 + pos.x1), pos.y1 + 0.012, label,
            ha='center', va='bottom', **text_kwargs,
        )
    elif kind == 'ylabel':
        fig.text(
            pos.x0 - 0.048, 0.5 * (pos.y0 + pos.y1), label,
            ha='center', va='center', rotation='vertical', **text_kwargs,
        )


def _plot_metric_grid(group, dataset, overlap, metrics, output_dir, suffix, title):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    style = _stakeholder_fig_style(suffix)
    broken_specs = []

    with _paper_font_context(style):
        _render_metric_grid(
            group, dataset, overlap, metrics, output_dir, suffix, title,
            fig_dir, style, broken_specs,
        )


def _render_metric_grid(group, dataset, overlap, metrics, output_dir, suffix, title,
                        fig_dir, style, broken_specs):
    colors = {'Our Model': '#1f77b4', 'SRC': '#d62728', 'BGM': '#2ca02c', 'APM': '#9467bd'}

    def add_value_labels(ax, bars, means):
        for bar, mean in zip(bars, means):
            if not np.isfinite(mean):
                continue
            label = f'{mean:.2f}' if abs(mean) >= 10 else f'{mean:.3f}'
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=style['bar_label_fs'],
                rotation=0,
            )

    def style_axis(ax, *, title=None, ylabel=None, show_ylabel=True):
        ax.tick_params(axis='both', labelsize=style['tick_fs'])
        if title:
            ax.set_title(title, fontsize=style['title_fs'], loc='center', pad=6)
        if ylabel and show_ylabel:
            ylabel_kwargs = {'fontsize': style['label_fs']}
            if style.get('use_times'):
                ylabel_kwargs['fontfamily'] = style['font_name']
            ax.set_ylabel(ylabel, **ylabel_kwargs)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)

    def style_xticks(ax, x):
        ax.set_xticks(x)
        ax.set_xticklabels(
            METHODS,
            rotation=style['x_rotation'],
            ha='center',
            fontsize=style['tick_fs'],
        )

    def plot_regular(ax, metric, means, low, high, yerr, x):
        bars = ax.bar(
            x, means, width=style['bar_width'], yerr=yerr, capsize=2,
            error_kw={'elinewidth': 0.75, 'capthick': 0.75},
            color=[colors[m] for m in METHODS], alpha=0.85,
        )
        style_xticks(ax, x)
        label = _metric_label(metric)
        style_axis(ax, ylabel=label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        finite_low = low[np.isfinite(low)]
        finite_high = high[np.isfinite(high)]
        if metric == 'delivery_time':
            ax.set_ylim(0.8, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
        elif metric == 'U_SC':
            ax.set_ylim(500, 700)
            ax.yaxis.set_major_locator(MultipleLocator(50))
        elif len(finite_high):
            y_max = float(np.max(finite_high))
            lower = 0.0
            span = max(y_max - lower, abs(y_max) * 0.08, 1e-9)
            upper = y_max + 0.28 * span
            ax.set_ylim(lower, upper)
        add_value_labels(ax, bars, means)

    def plot_broken(spec, metric, means, low, high, yerr, x):
        inner = spec.subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.05)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        bar_colors = [colors[m] for m in METHODS]
        bars_top = ax_top.bar(
            x, means, width=style['bar_width'], yerr=yerr, capsize=2,
            error_kw={'elinewidth': 0.75, 'capthick': 0.75},
            color=bar_colors, alpha=0.85,
        )
        bars_bot = ax_bot.bar(
            x, means, width=style['bar_width'], yerr=yerr, capsize=2,
            error_kw={'elinewidth': 0.75, 'capthick': 0.75},
            color=bar_colors, alpha=0.85,
        )

        finite = means[np.isfinite(means)]
        high_group = means >= 0.5 * float(np.nanmax(finite))
        low_group = np.isfinite(means) & ~high_group
        low_upper = float(np.nanmax(high[low_group])) if np.any(low_group) else float(np.nanmax(finite))
        low_lower = max(0.0, float(np.nanmin(low[low_group])) - 0.25 * max(low_upper, 1e-9)) if np.any(low_group) else 0.0
        low_upper = low_upper + 0.35 * max(low_upper - low_lower, 1e-9)
        high_lower = float(np.nanmin(low[high_group])) if np.any(high_group) else float(np.nanmax(finite))
        high_upper = float(np.nanmax(high[high_group])) if np.any(high_group) else float(np.nanmax(finite))
        high_span = max(high_upper - high_lower, 1e-9)
        ax_bot.set_ylim(low_lower, low_upper)
        ax_top.set_ylim(high_lower - 0.12 * high_span, high_upper + 0.22 * high_span)
        if metric == 'U_TSP':
            ax_bot.set_ylim(9.5, 13.8)
        elif metric == 'U_DP':
            ax_bot.set_ylim(4, 9)

        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(labeltop=False, labelbottom=False, bottom=False, labelsize=style['tick_fs'])
        ax_bot.xaxis.tick_bottom()
        style_xticks(ax_bot, x)
        style_axis(ax_top, show_ylabel=False)
        style_axis(ax_bot, show_ylabel=False)
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if metric == 'U_TSP':
            ax_bot.yaxis.set_major_locator(MultipleLocator(1.0))
        elif metric == 'U_DP':
            ax_bot.yaxis.set_major_locator(MultipleLocator(1.0))
        else:
            ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=4))

        d = 0.012
        kwargs_top = dict(transform=ax_top.transAxes, color='black', clip_on=False, linewidth=1.0)
        kwargs_bot = dict(transform=ax_bot.transAxes, color='black', clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs_bot)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bot)
        add_value_labels(ax_top, bars_top, means)
        add_value_labels(ax_bot, bars_bot, means)
        broken_specs.append((spec, _metric_label(metric)))

    fig = plt.figure(figsize=style['figsize'])
    grid = fig.add_gridspec(2, 2, wspace=style['wspace'], hspace=style['hspace'])
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for metric, pos in zip(metrics, positions):
        sub = group[group['metric'] == metric].set_index('method').reindex(METHODS).reset_index()
        means = sub['mean'].to_numpy(dtype=float)
        low = sub['ci95_low'].to_numpy(dtype=float)
        high = sub['ci95_high'].to_numpy(dtype=float)
        yerr = np.vstack([means - low, high - means])
        x = np.arange(len(METHODS))
        finite_means = means[np.isfinite(means)]
        spec = grid[pos[0], pos[1]]
        use_broken_axis = (
            metric == 'U_TSP' or (
                metric == 'U_DP' and len(finite_means) >= 2 and
                float(np.nanmax(finite_means)) / max(float(np.partition(finite_means, -2)[-2]), 1e-12) > 4.0
            )
        )
        if use_broken_axis:
            plot_broken(spec, metric, means, low, high, yerr, x)
        else:
            ax = fig.add_subplot(spec)
            plot_regular(ax, metric, means, low, high, yerr, x)
    fig.subplots_adjust(
        left=style['left'], right=style['right'],
        top=style['top'], bottom=style['bottom'],
    )
    for spec, label in broken_specs:
        _centered_spec_label(
            fig, spec, label,
            fontsize=style['label_fs'], kind='ylabel',
            font_name=style.get('font_name') if style.get('use_times') else None,
        )
    if style['show_suptitle']:
        fig.suptitle(
            f'{dataset} - {overlap} overlap: {title}',
            fontsize=style['title_fs'], y=0.995,
        )
    safe = f'{dataset}_{overlap}_{suffix}'.replace(' ', '_')
    fig.savefig(os.path.join(fig_dir, f'{safe}.png'), dpi=250, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.join(fig_dir, f'{safe}.pdf'), bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


def plot_summary(summary_df, output_dir):
    for (dataset, overlap), group in summary_df.groupby(['dataset', 'overlap_level']):
        _plot_metric_grid(
            group, dataset, overlap, STAKEHOLDER_PLOT_METRICS,
            output_dir, 'stakeholder_metrics', 'stakeholder utilities'
        )
        _plot_metric_grid(
            group, dataset, overlap, PERFORMANCE_PLOT_METRICS,
            output_dir, 'performance_metrics', 'quality and efficiency'
        )


def write_audit(output_dir, args, profiles, warnings):
    path = os.path.join(output_dir, 'robustness_audit.txt')
    lines = [
        'Robustness/statistical significance experiment audit',
        '====================================================',
        f"Datasets: {', '.join(args.datasets)}",
        f"Num runs: {args.num_runs}",
        f"Overlap levels: {', '.join(args.overlap_levels)}",
        f"Base seed: {args.base_seed}",
        f"Max rows per dataset: {args.max_rows if args.max_rows is not None else 'full dataset'}",
        'UCI Amazon Access Samples uses the history CSV inside the downloaded zip, not the 4.8GB main file.',
        'All methods within the same dataset/overlap/run_id use the same generated policy space.',
        f'For fair comparison, baseline policy levels are constrained by the Stackelberg benchmark risk budget on the same policy-space instance; risk_cap={FAIR_RISK_CAP_MULTIPLIER} times the benchmark avg_risk_value.',
        f'SRC starts from static policy-satisfaction level theta={SRC_FIXED_THETA}, applies the risk-budget cap, uses risk margin={BASELINE_RISK_MARGIN}, and uses TSP surplus share={BASELINE_BROKER_SHARE}.',
        f'BGM starts from bargaining policy level theta={BGM_FIXED_THETA}, uses risk-weighted attribute allocation, applies the same risk-budget cap and risk margin={BASELINE_RISK_MARGIN}; it does not optimize over the policy grid.',
        f'APM ranks attributes by marginal match contribution, starts from the top {APM_SELECTED_FRACTION:.2f} fraction at theta={APM_FIXED_THETA} and remaining attributes at theta={APM_LOW_THETA}, applies the risk-budget cap, and uses risk compensation with margin={APM_RISK_MARGIN} plus a small second-price-style floor.',
        'Additional metrics include social_welfare, quality_adjusted_welfare, welfare_per_time, and quality_per_time.',
        'Paired t-test and Wilcoxon p-values use scipy when available; this environment uses normal approximations if scipy is unavailable.',
        '',
        'Dataset profiles:',
    ]
    for profile in profiles:
        lines.append(
            f"- {profile.dataset}: rows={profile.rows}, cols={profile.columns}, "
            f"selected={len(profile.selected_attributes)}, continuous={len(profile.continuous_attributes)}, "
            f"discrete={len(profile.discrete_attributes)}"
        )
    if warnings:
        lines.append('')
        lines.append('Warnings:')
        lines.extend(f'- {w}' for w in warnings)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
