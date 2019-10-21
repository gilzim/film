import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn

if __name__ == '__main__':
    layer_index_to_plot = 0
    print("=== film_params load ===")
    film_params = np.load('../film_params.npy')
    film_params = film_params[:, layer_index_to_plot, :]

    print("=== q_types load ===")
    q_types = np.load('../q_types.npy')
    q_types = q_types.flatten()

    model = TSNE(n_components=2, perplexity=50, random_state=0)
    tsne_data = model.fit_transform(film_params)

    tsne_data = np.vstack((tsne_data.T, q_types)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("x", "y", "question_types"))
    g = sn.FacetGrid(tsne_df, hue="question_types", size=10)
    g.map(plt.scatter, 'x', 'y')
    legend = plt.legend(loc='best')
    labels = ['exist', 'less_than', 'greater_than',
              'count', 'query_material',
              'query_size', 'query_color',
              'query_shape', 'equal_color',
              'equal_integer', 'equal_shape',
              'equal_size', 'equal_material']
    for i, text in enumerate(legend.get_texts()):
        text.set_text(str(i) + " - " + labels[i])
    plt.show()
    print("Done Showing")


