"""
collect_sink_results.py

Parse lm_eval JSON output from Sink-Aware eval jobs and print comparison table.

lm_eval --output_path DIR saves results under:
  DIR/<model_label>/results_<timestamp>.json

Usage:
  python collect_sink_results.py
  python collect_sink_results.py --results_dir results/eval/sink_aware
"""
import os
import json
import glob
import argparse
from pathlib import Path

RESULTS_DIR = str(Path(__file__).resolve().parents[1] / 'results' / 'eval' / 'sink_aware')

# Task names as lm_eval records them in the JSON
TASK_KEYS = {
    'mmlu':          ('mmlu', 'acc,none'),
    'arc_challenge': ('arc_challenge', 'acc_norm,none'),
    'hellaswag':     ('hellaswag', 'acc_norm,none'),
}

# Display order
METHOD_ORDER = [
    'uncompressed',
    'wanda_50',
    'sink_50',
    'obs_diff',
    'tsvd_c4',      # TrajectSVD with C4-en calib (replaces old tsvd_mixed)
]

METHOD_LABELS = {
    'uncompressed': 'Uncompressed',
    'wanda_50':     'Wanda 50%',
    'sink_50':      'Sink-Aware 50%',
    'obs_diff':     'OBS-Diff 50%',
    'tsvd_c4':      'TrajectSVD (C4)',
}

PARAM_RETAIN = {
    'uncompressed': '100%',
    'wanda_50':     '50%',
    'sink_50':      '50%',
    'obs_diff':     '50%',
    'tsvd_c4':      '~80%',
}


def find_latest_result(model_dir, task):
    """Find the most recent results JSON for a given task inside model_dir."""
    task_dir = os.path.join(model_dir, task)
    if not os.path.isdir(task_dir):
        return None
    pattern = os.path.join(task_dir, '**', 'results*.json')
    files = glob.glob(pattern, recursive=True)
    if not files:
        # lm_eval sometimes saves directly in task_dir
        pattern2 = os.path.join(task_dir, 'results*.json')
        files = glob.glob(pattern2)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_score(results_json, task_key, metric_key):
    """Pull a metric value out of an lm_eval results JSON."""
    try:
        with open(results_json) as f:
            data = json.load(f)
        results = data.get('results', {})
        # Task may appear as exact key or with subtask suffix (e.g. mmlu_abstract_algebra…)
        if task_key in results:
            return results[task_key].get(metric_key)
        # Aggregate across subtasks for MMLU
        vals = [v.get(metric_key) for k, v in results.items()
                if k.startswith(task_key) and metric_key in v]
        if vals:
            return sum(vals) / len(vals)
    except Exception as e:
        print(f"  Warning: could not parse {results_json}: {e}")
    return None


def load_all_results(results_dir):
    """
    Scan results_dir for subdirs named <model_name>_<jobid> and
    collect scores for each (model, task).
    """
    scores = {}   # {model_name: {task: score}}

    if not os.path.isdir(results_dir):
        print(f"Results dir not found: {results_dir}")
        return scores

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # entry looks like "uncompressed_123456" or just "uncompressed"
        model_name = entry.rsplit('_', 1)[0] if '_' in entry else entry
        # Also accept plain name without job id
        if model_name not in METHOD_LABELS:
            # try the full entry
            model_name = entry
        if model_name not in METHOD_LABELS:
            continue

        if model_name not in scores:
            scores[model_name] = {}

        for task_display, (task_key, metric_key) in TASK_KEYS.items():
            if task_display in scores[model_name]:
                continue
            result_file = find_latest_result(entry_path, task_display)
            if result_file is None:
                continue
            val = extract_score(result_file, task_key, metric_key)
            if val is not None:
                scores[model_name][task_display] = val * 100   # convert to %

    return scores


def print_table(scores):
    col_w = [22, 12, 8, 8, 10, 6]
    header = ['Method', 'Param retain', 'MMLU', 'ARC-C', 'HellaSwag', 'Avg']
    sep = '+' + '+'.join('-' * w for w in col_w) + '+'

    def row(cells):
        return '|' + '|'.join(f' {c:<{w-1}}' for c, w in zip(cells, col_w)) + '|'

    print(sep)
    print(row(header))
    print(sep)

    tasks = ['mmlu', 'arc_challenge', 'hellaswag']
    for mkey in METHOD_ORDER:
        label  = METHOD_LABELS.get(mkey, mkey)
        retain = PARAM_RETAIN.get(mkey, '?')
        task_scores = scores.get(mkey, {})

        vals = [task_scores.get(t) for t in tasks]
        valid_vals = [v for v in vals if v is not None]
        avg = f'{sum(valid_vals)/len(valid_vals):.1f}' if valid_vals else '—'

        cells = [label, retain] + [f'{v:.1f}' if v is not None else '—' for v in vals] + [avg]
        print(row(cells))

    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = os.path.expanduser(args.results_dir)
    print(f"Scanning: {results_dir}\n")

    scores = load_all_results(results_dir)

    if not scores:
        print("No results found yet. Run eval jobs first.")
        return

    print_table(scores)

    # Also dump raw scores for inspection
    print("\nRaw scores (%):")
    for mkey in METHOD_ORDER:
        if mkey in scores:
            print(f"  {mkey}: {scores[mkey]}")


if __name__ == '__main__':
    main()
