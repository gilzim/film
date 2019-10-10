import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn

if __name__ == '__main__':
    print("=== film_params load ===")
    film_params = np.load('../film_params.npy')
    n, rows, cols = film_params.shape
    film_params = film_params.reshape((n, rows*cols))

    print("=== q_types load ===")
    q_types = np.load('../q_types.npy')
    q_types = q_types.flatten()

    print(film_params)
    print()
    print(q_types)

    print("Shapes", film_params.shape, q_types.shape)

    model = TSNE(n_components=2, perplexity=40, random_state=0)
    tsne_data = model.fit_transform(film_params)

    tsne_data = np.vstack((tsne_data.T, q_types)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("x", "y", "questions"))
    g = sn.FacetGrid(tsne_df, hue="questions", size=6)
    g.map(plt.scatter, 'x', 'y')
    plt.legend(loc='down right')
    plt.show()
    print("Done Showing")


