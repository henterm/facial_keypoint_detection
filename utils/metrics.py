import numpy as np


def compute_nme(pred, gt):
    """
    Normalized Mean Error normalized by bounding box size.
    pred, gt: np.array of shape (68, 2)
    """
    w = np.max(gt[:, 0]) - np.min(gt[:, 0])
    h = np.max(gt[:, 1]) - np.min(gt[:, 1])
    norm_factor = np.sqrt(w * h)
    if norm_factor < 1e-6:
        return None
    dist = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    return np.mean(dist) / norm_factor


def compute_auc(errors, error_thr=0.08):
    """
    AUC for CED curve, normalized over [0, error_thr] range.
    errors: np.array of sorted NME values
    """
    proportions = np.arange(errors.shape[0], dtype=np.float32) / errors.shape[0]
    auc = 0
    step = 0.01
    for thr in np.arange(0.0, 1.0, step):
        gt_indexes = [idx for idx, e in enumerate(errors) if e >= thr]
        if len(gt_indexes) > 0:
            first_gt_idx = gt_indexes[0]
        else:
            first_gt_idx = len(errors) - 1
        auc += proportions[first_gt_idx] * step
    return auc


def compute_ced(errors, error_thr=0.08):
    """
    Returns CED curve points clipped at error_thr.
    errors: np.array of NME values
    """
    errors = np.sort(errors)
    proportions = np.arange(1, len(errors) + 1) / len(errors)
    mask = errors <= error_thr
    return errors[mask], proportions[mask]