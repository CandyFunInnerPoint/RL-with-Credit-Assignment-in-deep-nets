import numpy as np

def get_batch_accuracy(out, target):

    out = out.round()
    wrong_preds = np.sum(np.absolute(out - target))
    acc = 1-wrong_preds/target.shape[0]

    return acc