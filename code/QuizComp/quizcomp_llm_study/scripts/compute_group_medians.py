#!/usr/bin/env python
import os, sys
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..', 'Results')
QUESTIONS = ['Q1_Accomplishment', 'Q2_Effort', 'Q3_Mental_Demand',
             'Q4_Controllability', 'Q5_Temporal_Demand', 'Q6_Satisfaction']
Q_SHORT = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']

CONFIGS = {
    'JudgeAgent': {
        'crowd': os.path.join(BASE, 'JudgeAgent', 'Results_as_csvs', 'crowd_survey.csv'),
        'GPT-5':  os.path.join(BASE, 'JudgeAgent', 'Results_as_csvs', 'GPT5', 'median_survey.csv'),
        'LLaMA-8B': os.path.join(BASE, 'JudgeAgent', 'Results_as_csvs', 'llama8B', 'median_survey.csv'),
        'Mistral-24B': os.path.join(BASE, 'JudgeAgent', 'Results_as_csvs', 'Mistral', 'median_survey.csv'),
    },
    'e2e Calibrated': {
        'crowd': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Calibrated', 'crowd_survey.csv'),
        'GPT-5':  os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Calibrated', 'GPT5', 'median_survey.csv'),
        'LLaMA-8B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Calibrated', 'llama8B', 'median_survey.csv'),
        'Mistral-24B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Calibrated', 'Mistral', 'median_survey.csv'),
    },
    'e2e Role-Anchored': {
        'crowd': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Role-Anchored', 'crowd_survey.csv'),
        'GPT-5':  os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Role-Anchored', 'GPT5', 'median_survey.csv'),
        'LLaMA-8B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Role-Anchored', 'llama8B', 'median_survey.csv'),
        'Mistral-24B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Role-Anchored', 'Mistral', 'median_survey.csv'),
    },
    'e2e Task-Constrained': {
        'crowd': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Task-Constrained', 'crowd_survey.csv'),
        'GPT-5':  os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Task-Constrained', 'GPT5', 'median_survey.csv'),
        'LLaMA-8B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Task-Constrained', 'llama8B', 'median_survey.csv'),
        'Mistral-24B': os.path.join(BASE, 'e2eAgent', 'Results_as_csvs', 'Task-Constrained', 'Mistral', 'median_survey.csv'),
    },
}

GROUP_ORDER = ['CROWD', 'GPT-5', 'LLaMA-8B', 'Mistral-24B']
ROLES = ['College Teacher', 'High School Teacher']


def load_and_median(path, questions, role=None):
    """Load CSV, optionally filter by role, return group median for each question."""
    df = pd.read_csv(path)
    if role is not None and 'Current_Job_Role' in df.columns:
        df = df[df['Current_Job_Role'] == role]
    return {q: float(np.median(df[q].dropna())) for q in questions}, len(df)

results = {}  # config -> group -> {q: median}
for config_name, paths in CONFIGS.items():
    results[config_name] = {}
    for group_key, path in paths.items():
        label = 'CROWD' if group_key == 'crowd' else group_key
        medians, _ = load_and_median(path, QUESTIONS)
        results[config_name][label] = medians

results_by_role = {}  # role -> config -> group -> {q: median}
sample_sizes = {}     # role -> config -> group -> n
for role in ROLES:
    results_by_role[role] = {}
    sample_sizes[role] = {}
    for config_name, paths in CONFIGS.items():
        results_by_role[role][config_name] = {}
        sample_sizes[role][config_name] = {}
        for group_key, path in paths.items():
            label = 'CROWD' if group_key == 'crowd' else group_key
            medians, n = load_and_median(path, QUESTIONS, role=role)
            results_by_role[role][config_name][label] = medians
            sample_sizes[role][config_name][label] = n

# Print results
sep = '-' * 80

print('=' * 80)
print('  GROUP MEDIANS  (median of per-persona medians over 5 runs)')
print('  CROWD values are raw single-run responses (no aggregation needed)')
print('=' * 80)

for config_name in CONFIGS:
    print()
    print(sep)
    print('  ' + config_name)
    print(sep)
    header = '{:<20s}'.format('Question')
    for g in GROUP_ORDER:
        header += ' | {:>12s}'.format(g)
    print(header)
    print(sep)
    for q, qs in zip(QUESTIONS, Q_SHORT):
        row = '{:<20s}'.format(qs + ' ' + q.split('_', 1)[1])
        for g in GROUP_ORDER:
            val = results[config_name][g][q]
            # Show as int if whole number, else 1 decimal
            if val == int(val):
                row += ' | {:>12d}'.format(int(val))
            else:
                row += ' | {:>12.1f}'.format(val)
        print(row)
    print()

# Combined comparison table
print()
print('=' * 80)
print('  PER-LLM COMPARISON ACROSS CONFIGS')
print('=' * 80)

for llm in ['GPT-5', 'LLaMA-8B', 'Mistral-24B']:
    print()
    print(sep)
    print('  ' + llm)
    print(sep)
    config_names = list(CONFIGS.keys())
    header = '{:<20s}'.format('Question')
    for cn in config_names:
        short = cn.replace('e2e ', '')
        header += ' | {:>16s}'.format(short)
    print(header)
    print(sep)
    for q, qs in zip(QUESTIONS, Q_SHORT):
        row = '{:<20s}'.format(qs + ' ' + q.split('_', 1)[1])
        for cn in config_names:
            val = results[cn][llm][q]
            if val == int(val):
                row += ' | {:>16d}'.format(int(val))
            else:
                row += ' | {:>16.1f}'.format(val)
        print(row)
    print()

# CROWD medians 
print()
print(sep)
print('  CROWD medians across configs')
print(sep)
config_names = list(CONFIGS.keys())
header = '{:<20s}'.format('Question')
for cn in config_names:
    short = cn.replace('e2e ', '')
    header += ' | {:>16s}'.format(short)
print(header)
print(sep)
for q, qs in zip(QUESTIONS, Q_SHORT):
    row = '{:<20s}'.format(qs + ' ' + q.split('_', 1)[1])
    for cn in config_names:
        val = results[cn]['CROWD'][q]
        if val == int(val):
            row += ' | {:>16d}'.format(int(val))
        else:
            row += ' | {:>16.1f}'.format(val)
    print(row)
print()

# Save CSV summary
rows = []
for config_name in CONFIGS:
    for g in GROUP_ORDER:
        for q, qs in zip(QUESTIONS, Q_SHORT):
            rows.append({
                'Config': config_name,
                'Group': g,
                'Role': 'All',
                'Question': qs + '_' + q.split('_', 1)[1],
                'Median': results[config_name][g][q]
            })
for role in ROLES:
    for config_name in CONFIGS:
        for g in GROUP_ORDER:
            for q, qs in zip(QUESTIONS, Q_SHORT):
                rows.append({
                    'Config': config_name,
                    'Group': g,
                    'Role': role,
                    'Question': qs + '_' + q.split('_', 1)[1],
                    'Median': results_by_role[role][config_name][g][q]
                })

out_csv = os.path.join(BASE, 'group_medians_summary.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print('Saved:', out_csv)

# Per role tables 
def fmt_val(val):
    if val == int(val):
        return '{:>12d}'.format(int(val))
    return '{:>12.1f}'.format(val)

for role in ROLES:
    print()
    print('=' * 80)
    print('  GROUP MEDIANS – ' + role)
    print('=' * 80)
    for config_name in CONFIGS:
        ns = sample_sizes[role][config_name]
        n_str = ', '.join('{}={}'.format(g, ns[g]) for g in GROUP_ORDER)
        print()
        print(sep)
        print('  {} (n: {})'.format(config_name, n_str))
        print(sep)
        header = '{:<20s}'.format('Question')
        for g in GROUP_ORDER:
            header += ' | {:>12s}'.format(g)
        print(header)
        print(sep)
        for q, qs in zip(QUESTIONS, Q_SHORT):
            row = '{:<20s}'.format(qs + ' ' + q.split('_', 1)[1])
            for g in GROUP_ORDER:
                row += ' | ' + fmt_val(results_by_role[role][config_name][g][q])
            print(row)
        print()

# Role comparison 
print()
print('=' * 80)
print('  ROLE COMPARISON (College Teacher vs High School Teacher)')
print('=' * 80)
for config_name in CONFIGS:
    print()
    print(sep)
    print('  ' + config_name)
    print(sep)
    header = '{:<20s}'.format('Question')
    for g in GROUP_ORDER:
        header += ' | {:>7s} {:>7s}'.format('CT', 'HST')
    print(header)
    print(sep)
    for q, qs in zip(QUESTIONS, Q_SHORT):
        row = '{:<20s}'.format(qs + ' ' + q.split('_', 1)[1])
        for g in GROUP_ORDER:
            ct = results_by_role['College Teacher'][config_name][g][q]
            hs = results_by_role['High School Teacher'][config_name][g][q]
            ct_s = '{:g}'.format(ct)
            hs_s = '{:g}'.format(hs)
            marker = '*' if ct != hs else ' '
            row += ' | {:>7s} {:>6s}{}'.format(ct_s, hs_s, marker)
        print(row)
    print()
    print('  * = median differs between College Teacher (CT) and High School Teacher (HST)')
