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
    film_params = np.load('film_params.npy')

    print("=== q_types load ===")
    q_types = np.load('q_types.npy')

    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(film_params)

    tsne_data = np.vstack((tsne_data.T, q_types)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim1", "Dim2", "label"))
    sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim1', 'Dim2')
    plt.show()


