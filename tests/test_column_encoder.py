import numpy as np
import time
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from column_encoder import MinHashEncoder


def test_MinHashEncoder(n_sample=70, minmax_hash=False):
    X_txt = fetch_20newsgroups(subset='train')['data']
    X = X_txt[:n_sample]

    for hashing in ['fast_hash', 'murmur_hash']:

        # Test output shape
        encoder = MinHashEncoder(n_components=50, hashing=hashing)
        encoder.fit(X)
        y = encoder.transform(X)
        assert y.shape == (n_sample, 50), str(y.shape)
        assert len(set(y[0])) == 50

        # Test same seed return the same output
        encoder = MinHashEncoder(50, hashing=hashing)
        encoder.fit(X)
        y2 = encoder.transform(X)
        np.testing.assert_array_equal(y, y2)

        # Test min property
        if not minmax_hash:
            X_substring = [x[:x.find(' ')] for x in X]
            encoder = MinHashEncoder(50, hashing=hashing)
            encoder.fit(X_substring)
            y_substring = encoder.transform(X_substring)
            np.testing.assert_array_less(y - y_substring, 0.0001)


def profile_encoder(Encoder, hashing='fast_hash'):

    from dirty_cat import datasets
    employee_salaries = datasets.fetch_employee_salaries()
    data = pd.read_csv(employee_salaries['path'])
    X = data['Employee Position Title']

    t0 = time.time()
    encoder = Encoder(n_components=50, hashing=hashing)
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (len(X), 50)
    return time.time() - t0


if __name__ == '__main__':
    print('start test')
    test_MinHashEncoder()
    print('test passed')

    print('time profile_encoder(MinHashEncoder, hashing=fast_hash)')
    print("{:.4} seconds".format(profile_encoder(MinHashEncoder, hashing='fast_hash')))
    print('time profile_encoder(MinHashEncoder, hashing=murmur_hash)')
    print("{:.4} seconds".format(profile_encoder(MinHashEncoder, hashing='murmur_hash')))

    print('Done')
