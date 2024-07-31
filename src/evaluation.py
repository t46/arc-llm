def eval_score(pred_grid, gt_grid, show=True):
    if pred_grid is None:
        score = -1.0
        return score

    pred_shape = pred_grid.shape
    gt_shape = gt_grid.shape
    if pred_shape != gt_shape:
        score = 0.0
    else:
        score = (pred_grid == gt_grid).sum() / (pred_shape[0] * pred_shape[1])

    if show:
        print("\033[93m" + f"score: {score}" + "\033[0m")
    return score