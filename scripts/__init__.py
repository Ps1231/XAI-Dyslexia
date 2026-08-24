"""XAI-Dyslexia pipeline scripts.

Package layout:
    scripts/common.py            shared utilities (fingerprints, state, fs helpers)
    scripts/data/                download -> extract -> organize datasets
    scripts/training/            dataset loaders + model training
    scripts/evaluation/          metrics, plots, reports
    scripts/pipeline.py          zero-config orchestrator

Run stages with:
    python -m scripts.pipeline              full pipeline
    python -m scripts.data.prepare          stage 1 only
    python -m scripts.training.train        stage 2 only
    python -m scripts.evaluation.evaluate   stage 3 only
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
