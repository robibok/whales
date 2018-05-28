import numpy as np
import pandas as pd
import sys


def normalize(x):
    return x / np.sum(x, axis=1, keepdims=True)


def norm_pow(preds, p):
    x = np.power(preds, p)
    return normalize(x)


def fix_mean(preds, il, right):
    left = normalize(np.sum(preds, axis=0, keepdims=True))
    return normalize(preds * norm_pow(right / left, il))


def add_eps(preds, eps):
    return normalize(preds + eps)


def main():
    assert(len(sys.argv) == 5)
    csv_paths = sys.argv[1:-1]
    output_name = sys.argv[-1]

    s = [pd.read_csv(path, header=0, index_col=0).sort_index() for path in csv_paths]
    s0 = s[0].copy()

    labels = pd.read_csv('trainLabels.csv', index_col=0, header=0)

    for i in xrange(3):
        assert(np.sum(s[0].index == s[i].index) == s[0].shape[0])


    aspiringto = np.zeros(shape=(1, 447))

    for i in xrange(labels.shape[0]):
        aspiringto[0, labels.level[i]] += 1

    right = normalize(aspiringto)

    to_merge = [(s[0], 2), (s[1], 1), (s[2], 1)]

    m = np.zeros((s0.shape[0], 447), dtype='float32')
    coef_sum = 0

    for (preds, coef) in to_merge:
        m += preds.values * coef
        coef_sum += coef

    m /= coef_sum

    final = fix_mean(norm_pow(add_eps(m, 0.0001), 1.5), 0.01, right)
    s0 *= 0
    s0 += final

    s0.to_csv(output_name)

    print 'DONE, results in {}'.format(output_name)


if __name__ == '__main__':
    main()
