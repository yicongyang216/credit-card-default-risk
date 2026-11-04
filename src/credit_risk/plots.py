"""Saved evaluation figures; consumes frozen predictions and never changes models."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from credit_risk.evaluation import metrics


def save_figures(run_dir, y, scores, validation_scores, validation_y, selected, threshold):
    folder = run_dir / 'figures'
    folder.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names = list(dict.fromkeys(['dummy', 'logistic', 'forest', selected]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name in names:
        RocCurveDisplay.from_predictions(y, scores[name], name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, scores[name], name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=.8)
    axes[1].axhline(np.mean(y), linestyle='--', color='grey', linewidth=.8)
    axes[0].set_title('Final test: ROC')
    axes[1].set_title('Final test: precision–recall (AP)')
    fig.savefig(folder / 'roc_pr.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name in names:
        observed, predicted = calibration_curve(y, scores[name], n_bins=10, strategy='quantile')
        axes[0].plot(predicted, observed, 'o-', label=name, markersize=4)
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set(xlabel='Mean predicted probability', ylabel='Observed default frequency', title='Test calibration: quantile bins')
    axes[0].legend(fontsize=8)
    axes[1].hist(scores[selected], bins=20, color='#396985')
    axes[1].set(xlabel='Predicted probability', ylabel='Test customers', title=f'Score distribution: {selected}')
    fig.savefig(folder / 'calibration.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, t, label in zip(axes, [.5, threshold], ['Default threshold 0.5', 'Frozen validation-capacity threshold']):
        ConfusionMatrixDisplay.from_predictions(y, scores[selected] >= t, labels=[0, 1],
            display_labels=['No default', 'Default'], ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(label)
    fig.savefig(folder / 'confusion_matrices.png', dpi=160)
    plt.close(fig)

    table = pd.read_csv(run_dir / 'validation_thresholds.csv')
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for name in ['precision', 'recall', 'flag_rate']:
        ax.plot(table['threshold'], table[name], label=name)
    ax.axvline(threshold, color='black', linestyle='--', label='Frozen threshold')
    ax.axhline(.2, color='grey', linestyle=':', label='Illustrative 20% flag budget')
    ax.set(xlabel='Probability threshold', ylabel='Rate', title='Validation threshold tradeoffs', ylim=(0, 1))
    ax.legend()
    fig.savefig(folder / 'threshold_tradeoffs.png', dpi=160)
    plt.close(fig)

    importance = pd.read_csv(run_dir / 'forest_validation_importance.csv').head(10).iloc[::-1]
    coefficients = pd.read_csv(run_dir / 'logistic_coefficients.csv').head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    axes[0].barh(importance['feature'], importance['ap_decrease'], xerr=importance['std'], color='#396985')
    axes[0].set(title='Forest: validation permutation importance', xlabel='AP decrease; 3-repeat standard deviation')
    labels = coefficients['feature'].str.replace('categorical__', '', regex=False).str.replace('numeric__', '', regex=False)
    axes[1].barh(labels, coefficients['coefficient'], color=['#c47842' if v > 0 else '#396985' for v in coefficients['coefficient']])
    axes[1].set(title='Logistic regression: largest absolute coefficients', xlabel='Coefficient; numeric variables standardized')
    fig.savefig(folder / 'feature_interpretation.png', dpi=160)
    plt.close(fig)
