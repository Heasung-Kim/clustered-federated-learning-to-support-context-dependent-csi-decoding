import numpy as np
from itertools import product, permutations
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import random
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from learning_agent.system_functions import flatten_tensor
from global_config import global_logger
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def spectral_clustering_and_matching(gradient_profile_matrix, n_centers=None, n_clients=None, estimated_cluster_ids_old=None):
    start_time = time.time()
    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    SVD_time = time.time() - start_time

    start_time = time.time()
    if n_centers is None:
        global_logger.info("singular_values:{}".format(singular_values))
        singular_gaps = singular_values[:-1] - singular_values[1:]
        global_logger.info("singular_gaps:{}".format(singular_gaps))
        num_of_leading_sv = 1 + np.argmax(singular_gaps)
        clipped_num_of_leading_sv = np.clip(num_of_leading_sv, a_min=3, a_max=int(n_clients / 1.5))
        global_logger.info("num of leading singular values / clipped: {}/{}".format(num_of_leading_sv, clipped_num_of_leading_sv))
        reduced_G = np.matmul(np.transpose(P[:, :clipped_num_of_leading_sv]), gradient_profile_matrix)
        silhouette_avg_list, estimated_part_ids_list = [], []
        max_n_clusters = int(n_clients / 1.5) if n_clients is not None else reduced_G.shape[1]
        for k in np.arange(2, max_n_clusters):
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42).fit(np.transpose(reduced_G))
            silhouette_avg = silhouette_score(np.transpose(reduced_G), kmeans.labels_)
            silhouette_avg_list.append(silhouette_avg)
            estimated_part_ids_list.append(kmeans.labels_)

        global_logger.info("silhouette_avg_list:{}".format(silhouette_avg_list))
        n_centers = np.argmax(silhouette_avg_list) + 2
        estimated_part_ids = estimated_part_ids_list[np.argmax(silhouette_avg_list)]

    else:
        reduced_G = np.matmul(np.transpose(P[:, :n_centers]), gradient_profile_matrix)
        kmeans = KMeans(n_clusters=n_centers, init="k-means++", random_state=42).fit(np.transpose(reduced_G))
        estimated_part_ids = kmeans.labels_
    K_means_clustering_time = time.time() - start_time

    start_time = time.time()
    if estimated_cluster_ids_old is not None:
        best_estimated_part_ids = estimated_part_ids
        best_n_consistency = 0
        for model_order in list(permutations([m_idx for m_idx in range(n_centers)])):
            temp_estimated_part_ids = np.array(model_order)[estimated_part_ids]
            n_consistency = np.sum((estimated_cluster_ids_old - temp_estimated_part_ids) == 0)
            if best_n_consistency < n_consistency:
                best_estimated_part_ids = temp_estimated_part_ids
                best_n_consistency = n_consistency
    else:
        best_estimated_part_ids = estimated_part_ids
    index_matching_time = time.time() - start_time

    info = {
        "estimated_cluster_ids": best_estimated_part_ids,
        "reduced_gradient_profile_matrix": reduced_G,
        "singular_values": singular_values,
        "SVD_time": SVD_time,
        "K_means_clustering_time": K_means_clustering_time,
        "index_matching_time": index_matching_time
    }
    return info



def get_num_cluster(gradient_profile_matrix, max_n_clusters, n_clients, estimated_cluster_ids_old=None):
    start_time = time.time()

    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    global_logger.info("singular_values:{}".format(singular_values))
    singular_gaps = singular_values[:-1] - singular_values[1:]
    global_logger.info("singular_gaps:{}".format(singular_gaps))
    num_of_leading_sv = 1 + np.argmax(singular_gaps)
    clipped_num_of_leading_sv = np.clip(num_of_leading_sv, a_min=3, a_max=n_clients)
    global_logger.info("num of leading singular values (min clipping 3) / clipped: {}/{}".format(num_of_leading_sv, clipped_num_of_leading_sv))

    reduced_G = np.matmul(np.transpose(P[:, :clipped_num_of_leading_sv]), gradient_profile_matrix)
    silhouette_avg_list = []
    for k in np.arange(2, max_n_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)  # .fit(np.transpose(reduced_G))
        cluster_labels = kmeans.fit_predict(np.transpose(reduced_G))
        silhouette_avg = silhouette_score(np.transpose(reduced_G), cluster_labels)
        silhouette_avg_list.append(silhouette_avg)

    global_logger.info("silhouette_avg_list:{}".format(silhouette_avg_list))
    proposed_k = np.argmax(silhouette_avg_list) + 2
    return proposed_k


def get_sparse_random_projection_matrix(n_features, compressed_n_features):
    positive_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)
    negative_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)
    zero_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)

    s = 3

    for i in range(compressed_n_features):
        #
        r = np.random.choice([1, 0, -1], size=n_features, p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])
        for j in range(n_features):
            if r[j] == 1:
                positive_matrix[i, j] = True
            elif r[j] == -1:
                negative_matrix[i, j] = True
            elif r[j] == 0:
                zero_matrix[i, j] = True
            else:
                raise AssertionError

    return positive_matrix, zero_matrix, negative_matrix


def get_sparse_projected_vector(positive_matrix, zero_matrix, negative_matrix, source):
    n_compressed_features, n_features = positive_matrix.shape
    target = np.zeros(shape=n_compressed_features)
    for i in range(n_compressed_features):
        target[i] = np.sum(source[positive_matrix[i]]) - np.sum(source[negative_matrix[i]])

    return target



#
# https://github.com/felisat/clustered-federated-learning
def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten_tensor(source1)
            s2 = flatten_tensor(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()


def compute_pairwise_similarities(clients):
    # dW = model_size_storage on model averaging scheme
    return pairwise_angles([client.model_size_storage for client in clients])


def cluster_clients(S):
    clustering = AgglomerativeClustering(metric="precomputed", linkage="complete").fit(-S)
    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    return c1, c2


def compute_max_update_norm(cluster):
    return np.max([torch.norm(flatten_tensor(client.model_size_storage)).item() for client in cluster])


def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([flatten_tensor(client.model_size_storage) for client in cluster]),
                                 dim=0)).item()


if __name__ == "__main__":
    # random data
    gradient_profile_matrix = np.random.random_sample(size=(128, 8))
    n_centers = 4
    n_clients = 8
    estimated_cluster_ids_old = None

    results = spectral_clustering_and_matching(gradient_profile_matrix,
                                               n_centers,
                                               n_clients,
                                               estimated_cluster_ids_old,
                                               clustering_algorithm="DBSCAN")
