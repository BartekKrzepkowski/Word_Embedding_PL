import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.decomposition import TruncatedSVD


def get_loaders(dataset, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loaders = {'train': loader}
    return loaders


def visualize_embeddings(embs, chosen_idxs, Ind2word):
    if len(chosen_idxs) != 0:
        embs = embs[chosen_idxs]
    temp = (embs - np.mean(embs, axis=0))
    covariance = 1.0 / len(temp) * (temp.T @ temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp @ U[:, :2]

    # svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
    # coord = svd.fit_transform(covariance)
    print('SVD done!')

    plt.figure(figsize=(15, 15))
    for i in range(len(chosen_idxs)):
        plt.text(coord[i, 0], coord[i, 1], Ind2word[chosen_idxs[i]],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'word_vectors_{stamp}.png')


def set_seeds(random_seed=42) -> None:
    """
    Set random seeds for reproduceability.
    See: https://pytorch.org/docs/stable/notes/randomness.html
         https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    """
    #########################
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    #########################
