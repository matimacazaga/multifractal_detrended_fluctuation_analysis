import numpy as np

from .fitting import polyeval, polyfit


def mf_dfa(x: np.ndarray, lags: np.ndarray, order: int, q:np.ndarray):

    y_profile = np.cumsum(x-x.mean())

    N = len(x)

    T = np.linspace(1, lags.max(), lags.max())

    q_ = q.reshape(-1,1)

    f = np.zeros(shape=(len(lags), len(q)))

    for i in range(len(lags)):

        lag = lags[i]

        y_segments = y_profile[:N-N%lag].reshape((N-N%lag)//lag, lag)

        y_segments_opposite = y_profile[N%lag:].reshape((N-N%lag)//lag, lag)

        coeffs = polyfit(T[:lag], y_segments.T, order)
        coeffs_opposite = polyfit(T[:lag], y_segments_opposite.T, order)

        F = np.append(
            np.var(y_segments - polyeval(coeffs, T[:lag]).T, axis=1),
            np.var(y_segments_opposite - polyeval(coeffs_opposite, T[:lag]).T, axis=1)
        )

        f[i, :] = np.float_power(np.mean(np.float_power(F, q_/2), axis=1), 1/q_.T)

    return f
