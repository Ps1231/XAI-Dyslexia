"""Stage 3 — evaluate trained models (idempotent, dynamic auto-capping).

Programmatic:
    from scripts.evaluation.evaluate import run_evaluation
    run_evaluation()                     # skips tasks already evaluated

CLI:
    python -m scripts.evaluation.evaluate [--task TASK] [--force]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.common import dir_fingerprint, load_state, save_state
from scripts.evaluation.plots import PLOTS_DIR
from scripts.evaluation.tasks import (
    evaluate_dyslexia_aggregate,
    evaluate_dyslexia_tabular,
    evaluate_image_task,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "app" / "models"
EVAL_STATE_DIR = PLOTS_DIR / ".state"

VALID_TASKS = ('dysgraphia', 'dyslexia_handwriting', 'dyslexia_synthetic',
               'dyslexia_tabular', 'dyslexia_aggregate', 'all')


def _eval_fingerprint(data_fp, model_dir, prefix):
    """Combine data fingerprint with this task's model-file stats."""
    h = hashlib.sha256()
    h.update((data_fp or "no-data").encode())
    for p in sorted(glob.glob(os.path.join(model_dir, f'{prefix}_*.pkl'))):
        h.update(f"{os.path.basename(p)}|{os.path.getsize(p)}\n".encode())
    return h.hexdigest()


def _eval_up_to_date(task_key, fp, force):
    if force or not fp:
        return False
    prev = load_state(EVAL_STATE_DIR, f"eval_{task_key}")
    if prev and prev.get("fingerprint") == fp:
        print(f"\n[skip] {task_key}: models & data unchanged since last evaluation")
        return True
    return False


def _mark_eval_done(task_key, fp):
    if fp:
        save_state(EVAL_STATE_DIR, f"eval_{task_key}", {"fingerprint": fp})


def run_evaluation(
    task: str = "all",
    data_dir=None,
    model_dir=None,
    max_images: int | None = None,
    force: bool = False,
):
    """Evaluate trained models — skipping tasks already evaluated on identical inputs.

    Args:
        task: One of VALID_TASKS.
        data_dir: Organized-data dir (default <project>/data/processed).
        model_dir: Trained-model dir (default <project>/app/models).
        max_images: Per-class cap for image tasks; None = dynamic auto-cap.
        force: Re-evaluate even when nothing changed.

    Returns:
        List of per-model report dicts for THIS run (the merged history is
        written to evaluation_plots/master_evaluation_report.json).
    """
    data_dir = str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
    model_dir = str(Path(model_dir) if model_dir else DEFAULT_MODEL_DIR)

    all_results = []

    def _do_image(t):
        src = os.path.join(data_dir, t)
        fp = _eval_fingerprint(dir_fingerprint(src), model_dir, t)
        if _eval_up_to_date(t, fp, force):
            return
        cap = resolve_cap(t, src)
        results = evaluate_image_task(t, data_dir, model_dir, max_per_class=cap)
        if results:
            _mark_eval_done(t, fp)
        all_results.extend(results)

    def resolve_cap(t, src):
        if max_images is not None:
            return max_images
        from scripts.evaluation.tasks import resolve_auto_cap
        return resolve_auto_cap(src)

    def _do_tabular(t, fn):
        src = os.path.join(data_dir, 'tabular')
        fp = _eval_fingerprint(dir_fingerprint(src), model_dir, t)
        if _eval_up_to_date(t, fp, force):
            return
        results = fn(data_dir, model_dir)
        if results:
            _mark_eval_done(t, fp)
        all_results.extend(results)

    if task in ('dysgraphia', 'all'):
        _do_image('dysgraphia')
    if task in ('dyslexia_handwriting', 'all'):
        _do_image('dyslexia_handwriting')
    if task in ('dyslexia_synthetic', 'all'):
        _do_image('dyslexia_synthetic')
    if task in ('dyslexia_tabular', 'all'):
        _do_tabular('dyslexia_tabular', evaluate_dyslexia_tabular)
    if task in ('dyslexia_aggregate', 'all'):
        _do_tabular('dyslexia_aggregate', evaluate_dyslexia_aggregate)

    # Save master report (merge with previous so skips don't erase history)
    report_path = PLOTS_DIR / 'master_evaluation_report.json'
    seen = {(r['task'], r['model']) for r in all_results}
    merged = []
    if report_path.exists():
        try:
            for r in json.loads(report_path.read_text(encoding='utf-8')):
                if (r.get('task'), r.get('model')) not in seen:
                    merged.append(r)
        except Exception:
            pass
    merged.extend(all_results)
    with open(report_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Models evaluated this run: {len(all_results)} ({len(merged)} total)")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Master report: {report_path}")

    if merged:
        print(f"\n{'Model':<40} {'Task':<20} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7}")
        print("-" * 100)
        for r in merged:
            auc_str = f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else "N/A"
            sens_str = f"{r['sensitivity']:.3f}" if r['sensitivity'] is not None else "N/A"
            spec_str = f"{r['specificity']:.3f}" if r['specificity'] is not None else "N/A"
            print(f"{r['model']:<40} {r['task']:<20} {r['accuracy']:>7.3f} "
                  f"{r['f1']:>7.3f} {sens_str:>7} {spec_str:>7} {auc_str:>7}")

    return all_results


def main():
    """Optional CLI shim."""
    parser = argparse.ArgumentParser(description='Evaluate all trained XAI-Dyslexia models')
    parser.add_argument('--task', choices=list(VALID_TASKS), default='all')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images per class for image tasks (default: dynamic auto-cap)')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even when nothing changed')
    args = parser.parse_args()

    run_evaluation(
        task=args.task,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        max_images=args.max_images,
        force=args.force,
    )


if __name__ == '__main__':
    main()
