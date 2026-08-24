"""Matplotlib plot helpers for evaluation (confusion matrix, ROC, distributions)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation_plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Save a confusion matrix heatmap."""
    cm = np.array(cm)  # handle list input from JSON serialization
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, title, save_path):
    """Save ROC curve for binary classification. Returns the AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return auc


def plot_class_distribution(y, class_names, title, save_path):
    """Bar chart of class distribution."""
    counts = np.bincount(y, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts,
                  color=['#0891b2', '#ef4444', '#f59e0b', '#10b981'][:len(class_names)])
    ax.set_title(title)
    ax.set_ylabel('Count')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(c), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
