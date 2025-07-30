import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Wczytaj dane
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Dodaj kolumnę z nazwami gatunków
df['species_name'] = [iris.target_names[i] for i in iris.target]

# Lista wszystkich par cech
features = iris.feature_names
n_features = len(features)

# Tworzymy wykresy dla każdej pary cech
for i in range(n_features):
    for j in range(i+1, n_features):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x=features[i],
            y=features[j],
            hue='species_name',  # <-- używamy nazw gatunków
            palette='Set1',
            s=70
        )
        plt.title(f'Iris: rzut na cechy ({features[i]} vs {features[j]})')
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.legend(title='Gatunek')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
