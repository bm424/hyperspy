import numpy as np
import tqdm


def try_again(x, q, algorithm, attempts, verbose=False, **kwargs):
    j_best = np.infty
    n = 0
    try_over = range(attempts)
    if verbose:
        try_over = tqdm.tqdm(try_over)
    for n in try_over:
        trial = algorithm(x, q, **kwargs).optimize()
        if trial.J < j_best:
            best_trial = trial
            j_best = trial.J
            if verbose:
                try_over.write(
                    "Trial {}/{}: j_best = {:.2f}".format(n, attempts, j_best))
        elif np.isnan(trial.J):
            continue
    return best_trial
