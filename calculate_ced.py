import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from utils.metrics import compute_nme, compute_auc, compute_ced


def load_pts(pts_path):
    return np.loadtxt(pts_path, comments=("version:", "n_points:", "{", "}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Directory with ground truth .pts files')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Directory with predicted .pts files')
    parser.add_argument('--output_path', type=str, default='ced.png')
    parser.add_argument('--error_thr', type=float, default=0.08)
    args = parser.parse_args()

    errors = []
    gt_files = glob(os.path.join(args.gt_path, '*.pts'))

    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        pred_path = os.path.join(args.pred_path, fname)

        if not os.path.exists(pred_path):
            continue

        gt = load_pts(gt_path)
        pred = load_pts(pred_path)
        errors.append(compute_nme(pred, gt))

    errors = np.array(errors)
    auc = compute_auc(errors, args.error_thr)
    ced_x, ced_y = compute_ced(errors, args.error_thr)

    print(f"Samples evaluated: {len(errors)}")
    print(f"Mean NME: {np.mean(errors):.4f}")
    print(f"AUC @ {args.error_thr}: {auc:.3f}")

    plt.figure(figsize=(10, 7))
    plt.plot(ced_x, ced_y, linewidth=2.0,
             label=f'Our model, auc={auc:.3f}')
    plt.xlabel('NME')
    plt.ylabel('Proportion of images')
    plt.title('Cumulative Error Distribution')
    plt.legend()
    plt.savefig(args.output_path)
    print(f"CED plot saved to {args.output_path}")


if __name__ == '__main__':
    main()