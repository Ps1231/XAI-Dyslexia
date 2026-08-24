"""
One-command pipeline for the XAI-Dyslexia project.

Runs every stage in order, skipping any stage whose inputs are unchanged:

    1. Prepare    download -> extract -> organize datasets   (data/processed/)
    2. Train      train + save models per task               (app/models/)
    3. Evaluate   metrics + plots per model                  (evaluation_plots/)

Zero-argument usage:

    python -m scripts.pipeline

Programmatic usage:

    from scripts.pipeline import run_pipeline

    run_pipeline()                       # full pipeline, skip what's done
    run_pipeline(force=True)             # redo everything
    run_pipeline(task="dysgraphia")      # single task end-to-end
    run_pipeline(evaluate=False)         # stop after training
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

from scripts.data.prepare import run_prepare
from scripts.training.train import VALID_TASKS, run_training
from scripts.evaluation.evaluate import run_evaluation

CONSOLE = Console()


def run_pipeline(
    task: str = "all",
    force: bool = False,
    evaluate: bool = True,
) -> dict:
    """Run prepare -> train -> (optionally) evaluate, skipping finished stages.

    Args:
        task: 'all' or one of 'dysgraphia', 'dyslexia_tabular',
            'dyslexia_handwriting', 'dyslexia_aggregate'.
        force: Ignore all up-to-date markers and redo every stage.
        evaluate: Run the evaluation stage after training.

    Returns:
        Dict with per-stage summaries.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {', '.join(VALID_TASKS)}")

    summary = {}

    CONSOLE.rule("[bold cyan]STAGE 1/3 - DATA PREPARATION")
    summary["prepare"] = run_prepare(force=force)
    CONSOLE.print(summary["prepare"])

    CONSOLE.rule("[bold cyan]STAGE 2/3 - MODEL TRAINING")
    run_training(task=task, force=force)

    if evaluate:
        CONSOLE.rule("[bold cyan]STAGE 3/3 - EVALUATION")
        summary["evaluation"] = run_evaluation(task=task, force=force)

    CONSOLE.rule("[bold green]PIPELINE COMPLETE")
    CONSOLE.print(
        "[green]Models are ready. Launch the web app with:[/] [bold]python run.py[/]"
    )
    return summary


def main():
    """Optional CLI shim."""
    import argparse

    parser = argparse.ArgumentParser(description="Full XAI-Dyslexia pipeline")
    parser.add_argument("--task", default="all", choices=list(VALID_TASKS))
    parser.add_argument("--force", action="store_true", help="Redo every stage")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    run_pipeline(task=args.task, force=args.force, evaluate=not args.no_evaluate)


if __name__ == "__main__":
    main()
