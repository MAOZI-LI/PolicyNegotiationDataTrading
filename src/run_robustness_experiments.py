"""Run repeated robustness experiments with confidence intervals and tests."""

import argparse
import os

import pandas as pd

from robustness_utils import (
    DATASET_CONFIGS,
    METHODS,
    generate_policy_space,
    load_dataset,
    plot_summary,
    profile_dataset,
    profile_to_row,
    run_method,
    run_significance_tests,
    save_latex_tables,
    summarize_results,
    write_audit,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Repeated robustness/statistical significance experiments.')
    parser.add_argument('--datasets', nargs='+', default=['amazon_employee', 'uci_amazon_access', 'incident_event_log'])
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--overlap-levels', nargs='+', default=['low', 'medium', 'high'],
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--base-seed', type=int, default=2026)
    parser.add_argument('--output-dir', default='results/statistical_robustness')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Optional row cap for quick profiling/debugging. Default reads full configured data.')
    return parser.parse_args()


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)


def run(args):
    ensure_output_dir(args.output_dir)
    profiles = []
    profile_rows = []
    raw_rows = []
    warnings = []

    for dataset_name in args.datasets:
        print(f'[dataset] loading {dataset_name}', flush=True)
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f'unknown dataset {dataset_name}; known datasets: {sorted(DATASET_CONFIGS)}')
        config = DATASET_CONFIGS[dataset_name]
        path = config['path']
        df_raw = load_dataset(dataset_name, path=path, config=config, nrows=args.max_rows)
        df, profile = profile_dataset(dataset_name, df_raw, path, config=config)
        print(
            f'[dataset] {dataset_name}: rows={len(df_raw)}, selected={len(profile.selected_attributes)}, '
            f'continuous={len(profile.continuous_attributes)}, discrete={len(profile.discrete_attributes)}',
            flush=True,
        )
        profiles.append(profile)
        profile_rows.append(profile_to_row(profile))
        if not profile.continuous_attributes:
            warnings.append(f'{dataset_name}: no continuous attributes selected; experiment continues with discrete attributes only.')
        if not profile.discrete_attributes:
            warnings.append(f'{dataset_name}: no discrete attributes selected; experiment continues with continuous attributes only.')

        for overlap_level in args.overlap_levels:
            print(f'[run] {dataset_name} overlap={overlap_level}', flush=True)
            for run_id in range(args.num_runs):
                seed = args.base_seed + run_id
                try:
                    policy_space = generate_policy_space(dataset_name, df, profile, overlap_level, seed)
                except Exception as exc:
                    warnings.append(f'{dataset_name}/{overlap_level}/run{run_id}: policy-space generation failed: {exc}')
                    continue
                for method in METHODS:
                    print(f'  [method] run={run_id} seed={seed} method={method}', flush=True)
                    row = run_method(
                        method, dataset_name, overlap_level, run_id, seed,
                        policy_space, data_size=len(df_raw),
                    )
                    raw_rows.append(row)

    profiles_df = pd.DataFrame(profile_rows)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_results(raw_df)
    tests_df, test_warnings = run_significance_tests(raw_df)
    warnings.extend(test_warnings)

    profiles_df.to_csv(os.path.join(args.output_dir, 'dataset_profiles.csv'), index=False)
    raw_df.to_csv(os.path.join(args.output_dir, 'robustness_raw_results.csv'), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, 'robustness_summary.csv'), index=False)
    tests_df.to_csv(os.path.join(args.output_dir, 'significance_tests.csv'), index=False)
    save_latex_tables(summary_df, tests_df, args.output_dir)
    plot_summary(summary_df, args.output_dir)
    write_audit(args.output_dir, args, profiles, warnings)

    return profiles_df, raw_df, summary_df, tests_df


def main():
    args = parse_args()
    profiles_df, raw_df, summary_df, tests_df = run(args)
    print(f'[ok] dataset profiles: {len(profiles_df)} rows')
    print(f'[ok] raw results: {len(raw_df)} rows')
    print(f'[ok] summary: {len(summary_df)} rows')
    print(f'[ok] significance tests: {len(tests_df)} rows')
    print(f'[ok] output dir: {args.output_dir}')


if __name__ == '__main__':
    main()
