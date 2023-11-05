import torch


def initialize_cluster(grouping_method, total_clusters, num_timesteps=1000):
    if grouping_method == "uniform":
        clusters = []
        print("uniform clustering")
        for cluster_ind in range(total_clusters):
            min_t = int(num_timesteps / total_clusters * cluster_ind)
            max_t = int(num_timesteps / total_clusters * (cluster_ind + 1))
            clusters.append((min_t, max_t))
    else:
        raise NotImplementedError(f"{grouping_method} is not supported for grouping")
    return clusters


def sample_t_batch(batch_size, clusters, device):
    total_clusters = len(clusters)
    t_list = []
    for min_t, max_t in clusters:
        t = torch.randint(min_t, max_t, (int(batch_size / total_clusters),), device=device)
        t_list.append(t)
    return torch.cat(t_list, dim=0)
