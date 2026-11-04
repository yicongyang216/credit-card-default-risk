"""Evaluation conventions shared by the runner, notebooks and tests."""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, precision_score, recall_score, f1_score)


def capacity_threshold(scores, capacity=0.2) -> float:
    """Flag highest score groups within a validation budget; never split ties."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Scores must be a nonempty finite vector')
    if ((scores < 0) | (scores > 1)).any() or not 0 <= capacity <= 1:
        raise ValueError('Probabilities and capacity must lie in [0, 1]')
    k = int(np.floor(capacity * len(scores)))
    if k == 0:
        return float(np.nextafter(scores.max(), np.inf))
    cutoff = np.sort(scores)[::-1][k-1]
    if (scores >= cutoff).sum() > k:
        cutoff = np.nextafter(cutoff, np.inf)
    return float(cutoff)


def metrics(y, scores, threshold=0.5) -> dict:
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, scores)),
        'average_precision': float(average_precision_score(y, scores)),
        'brier': float(brier_score_loss(y, scores)), 'log_loss': float(log_loss(y, scores)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'flag_rate': float(predicted.mean()), 'threshold': float(threshold),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'n': len(y),
        'prevalence': float(y.mean())}


def subgroup_metrics(y, scores, groups, threshold, minimum=20) -> list[dict]:
    """Suppress a rate if its denominator is <minimum; keep raw counts visible."""
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups).astype(str)
    result = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        actual, flagged = y[mask], scores[mask] >= threshold
        tn, fp, fn, tp = confusion_matrix(actual, flagged, labels=[0, 1]).ravel()
        positives, negatives, selected = int(tp+fn), int(tn+fp), int(tp+fp)
        result.append({'group': group, 'n': int(mask.sum()), 'positives': positives,
            'negatives': negatives, 'flagged': selected, 'prevalence': float(actual.mean()),
            'flag_rate': float(flagged.mean()), 'recall': float(tp/positives) if positives >= minimum else None,
            'fpr': float(fp/negatives) if negatives >= minimum else None,
            'precision': float(tp/selected) if selected >= minimum else None,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return result


def load_partitions(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if (not manifest['ID'].is_unique or not frame['ID'].is_unique
        or set(manifest['ID']) != set(frame['ID'])
        or set(manifest['split']) != {'train', 'validation', 'test'}):
        raise ValueError('Invalid split identities or partition labels')
    assigned = frame.merge(manifest, on='ID', validate='one_to_one').sort_values('ID')
    return {name: assigned.loc[assigned['split'] == name].drop(columns='split').reset_index(drop=True)
            for name in ['train', 'validation', 'test']}
