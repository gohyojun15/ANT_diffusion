import numpy as np

# SNR-based interval clustering.
def snr_based_clustering(snr, k):
    """
    Produce clustering results from SNR-based interval clustering.

    :param snr: contains snr values corresponding to diffusion timestep as np.array([diffusion_timesteps])
    :param k: the number of clusters
    :param min_size: minimum size of clusters
    :param max_size: max size of clusters
    :return:
    """
    n = snr.shape[0]
    D_matrix = np.zeros((n, k))
    S_matrix = np.zeros((n, k))

    def calculate_interval_loss(left, right):
        if left == right:
            return 0
        else:
            center = round((left + right + 1) / 2)
            score = 0

            score += np.abs(snr[left : right + 1] - snr[center]).sum()

        return score

    for j in range(k):
        for i in reversed(range(n)):
            if j == 0:
                D_matrix[i, j] = calculate_interval_loss(0, i)
            else:
                if i >= j:
                    items = np.zeros(i)
                    for L in range(j, i):
                        items[L] = D_matrix[L, j - 1] + calculate_interval_loss(L + 1, i)
                    items[items == 0] = np.inf

                    D_matrix[i, j] = items.min()
                    S_matrix[i, j] = items.argmin()

    # Back track
    bounds = []
    b = 0
    for j in reversed(range(k)):
        if j == k - 1:
            b = S_matrix[-1, k - 1]
        else:
            b = S_matrix[int(b), j]
        bounds.append(b)

    # produce final clusters
    clusters = []
    for k_ind in range(k):
        left = list(reversed(bounds))[k_ind]
        right = list(reversed(bounds))[k_ind + 1] - 1 if (k_ind + 1) < k else 1000
        clusters.append((left, right))

    return clusters


def tas_based_clustering(tas, k):
    n = tas.shape[0]
    D_matrix = np.zeros((n, k)) + np.inf
    S_matrix = np.zeros((n, k))

    def calculate_interval_loss(left, right):
        if left == right:
            return -1.0
        else:
            center = int((left + right + 1) / 2)
            score = 0
            # for ind in range(left, right + 1):
            score -= tas[center, left : right + 1].sum()
            return score

    for j in range(k):
        for i in reversed(range(n)):
            if j == 0:
                D_matrix[i, j] = calculate_interval_loss(0, i)
            else:
                if i >= j:
                    items = np.zeros(i)
                    for L in range(j, i):
                        items[L] = D_matrix[L, j - 1] + calculate_interval_loss(L + 1, i)
                    items[items == 0] = np.inf

                    D_matrix[i, j] = items.min()
                    S_matrix[i, j] = items.argmin()
    # Back track
    bounds = []
    b = 0
    for j in reversed(range(k)):
        if j == k - 1:
            b = S_matrix[-1, k - 1]
        else:
            b = S_matrix[int(b), j]
        bounds.append(b)

    # produce final clusters
    clusters = []
    for k_ind in range(k):
        left = list(reversed(bounds))[k_ind]
        right = list(reversed(bounds))[k_ind + 1] - 1 if (k_ind + 1) < k else 1000
        clusters.append((left, right))
    return clusters
