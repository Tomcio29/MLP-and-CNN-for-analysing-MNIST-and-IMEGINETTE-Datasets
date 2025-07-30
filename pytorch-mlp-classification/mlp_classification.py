import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.decomposition import PCA
import itertools
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

def four_quadrants(img):
    q1 = img[:14, :14].sum()
    q2 = img[:14, 14:].sum()
    q3 = img[14:, :14].sum()
    q4 = img[14:, 14:].sum()
    return [q1, q2, q3, q4]

# --- Załaduj dane ---
transform = transforms.ToTensor()
mnist_train = MNIST(root='.', train=True, download=True, transform=transform)

# --- Fit PCA na całym zbiorze ---
X_train = []
for img, _ in mnist_train:
    img_np = np.array(img, dtype=np.float32).flatten()
    X_train.append(img_np)
X_train = np.stack(X_train)
pca = PCA(n_components=2)
pca.fit(X_train)

# --- Znajdź pierwsze wystąpienie każdej cyfry 0-9 ---
digits = list(range(10))
results = []
images = []

for digit in digits:
    for idx, (img, label) in enumerate(mnist_train):
        if label == digit:
            img_np = np.array(img, dtype=np.float32)
            if img_np.shape == (1, 28, 28):
                img_np = img_np[0]
            quads = four_quadrants(img_np)
            pca_feats = pca.transform(img_np.flatten().reshape(1, -1)).flatten()
            results.append({
                'Digit': digit,
                'Q1': quads[0],
                'Q2': quads[1],
                'Q3': quads[2],
                'Q4': quads[3],
                'PCA1': pca_feats[0],
                'PCA2': pca_feats[1]
            })
            images.append(img_np)
            plt.imsave(f"mnist_digit_{digit}.png", img_np, cmap='gray')
            break

df = pd.DataFrame(results)
df['PCA1'] = df['PCA1'].round(2)
df['PCA2'] = df['PCA2'].round(2)

# --- Przygotuj dane do PCA ---
X = []
y = []
X_quad = []  # Dodane
for img, label in mnist_train:
    img_np = np.array(img, dtype=np.float32).flatten()
    X.append(img_np)
    y.append(label)

    # Dodane: ekstrakcja quadrants dla wszystkich obrazów
    img_2d = np.array(img, dtype=np.float32).squeeze()  # Usuń wymiar kanału
    quads = four_quadrants(img_2d)
    X_quad.append(quads)

X = np.stack(X)
y = np.array(y)
X_quad = np.stack(X_quad)  # Dodane

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# -- 1. Uniwersalna klasa Dataset dla scikit-learn i MNIST (po ekstrakcji cech) --
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -- 2. Ekstraktory cech dla MNIST --
def flatten_features(img):
    return img.flatten()

def two_features_top_bottom(img):
    # Suma pikseli w górnej i dolnej połowie
    top = img[:14, :].sum()
    bottom = img[14:, :].sum()
    return np.array([top, bottom])

def two_features_left_right(img):
    # Suma pikseli w lewej i prawej połowie
    left = img[:, :14].sum()
    right = img[:, 14:].sum()
    return np.array([left, right])

def four_quadrants(img):
    # Suma pikseli w 4 ćwiartkach
    q1 = img[:14, :14].sum()
    q2 = img[:14, 14:].sum()
    q3 = img[14:, :14].sum()
    q4 = img[14:, 14:].sum()
    return np.array([q1, q2, q3, q4])

def center_of_mass(img):
    # Obsługa przypadku (1, 28, 28)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    assert img.ndim == 2 and img.shape == (28, 28), f"Expected 2D image, got shape {img.shape}"
    y, x = np.indices(img.shape)
    total = img.sum()
    if total == 0:
        return np.array([0, 0])
    x_mean = (x * img).sum() / total
    y_mean = (y * img).sum() / total
    return np.array([x_mean, y_mean])

def features(img):
    # Example: 6 features: top, bottom, left, right sums + center of mass (2)
    top = img[:14, :].sum()
    bottom = img[14:, :].sum()
    left = img[:, :14].sum()
    right = img[:, 14:].sum()
    com = center_of_mass(img)
    return np.array([top, bottom, left, right, com[0], com[1]])

def fit_pca_on_mnist(n_components=2):
    # Wczytaj MNIST, spłaszcz obrazy
    transform = transforms.ToTensor()
    mnist_train = MNIST(root='.', train=True, download=True, transform=transform)
    X_train = []
    for img, _ in mnist_train:
        img_np = np.array(img, dtype=np.float32).flatten()
        X_train.append(img_np)
    X_train = np.stack(X_train)
    # Wytrenuj PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return pca

def pca_features(img, pca):
    flat = img.flatten().reshape(1, -1)
    return pca.transform(flat).flatten()

# -- 3. Funkcja do przetwarzania MNIST na cechy --
def mnist_to_features(mnist_dataset, feature_fn, **kwargs):
    X = []
    y = []
    for img, label in mnist_dataset:
        img_np = np.array(img, dtype=np.float32)
        features = feature_fn(img_np, **kwargs) if kwargs else feature_fn(img_np)
        X.append(features)
        y.append(label)
    X = np.stack(X)
    y = np.array(y)
    return X, y

# -- 4. MLP: dwie warstwy Linear, opcjonalnie ReLU --
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_relu=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU() if use_relu else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# -- 5. Funkcja treningowa --
def train_model(model, train_loader, test_loader, epochs=20, lr=0.01, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        # Ewaluacja
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_true.extend(y_batch.numpy())
                y_pred.extend(preds)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
        print(f"Epoch {epoch+1}/{epochs}, test accuracy: {acc:.4f}")
    # Przywróć najlepszy model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -- 6. Funkcja do ewaluacji i raportu --
def evaluate_model(model, loader, device='cpu'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm

# -- 7. Przygotowanie danych scikit-learn --
def prepare_sklearn_dataset(load_fn, test_size=0.2, random_state=42):
    data = load_fn()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state, stratify=data.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return NumpyDataset(X_train, y_train), NumpyDataset(X_test, y_test), data.target_names

# -- 8. Przygotowanie MNIST --
def prepare_mnist(feature_fn, batch_size=128, pca=None, **kwargs):
    transform = transforms.ToTensor()
    mnist_train = MNIST(root='.', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='.', train=False, download=True, transform=transform)
    # Jeśli feature_fn to pca_features, przekaż obiekt PCA
    if feature_fn == pca_features and pca is not None:
        X_train, y_train = mnist_to_features(mnist_train, feature_fn, pca=pca)
        X_test, y_test = mnist_to_features(mnist_test, feature_fn, pca=pca)
    else:
        X_train, y_train = mnist_to_features(mnist_train, feature_fn, **kwargs)
        X_test, y_test = mnist_to_features(mnist_test, feature_fn, **kwargs)
    train_ds = NumpyDataset(X_train, y_train)
    test_ds = NumpyDataset(X_test, y_test)
    return train_ds, test_ds

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)

    # --- Iris ---
    train_ds, test_ds, class_names = prepare_sklearn_dataset(load_iris)
    model = SimpleMLP(input_dim=4, hidden_dim=16, output_dim=3)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    model = train_model(model, train_loader, test_loader, epochs=30, lr=0.05, device=device)

    # Ewaluacja na obu zbiorach
    acc_train, cm_train = evaluate_model(model, train_loader, device)
    acc_test, cm_test = evaluate_model(model, test_loader, device)
    print("Iris train accuracy:", acc_train)
    print("Iris train confusion matrix:\n", cm_train)
    print("Iris test accuracy:", acc_test)
    print("Iris test confusion matrix:\n", cm_test)
    torch.save(model.state_dict(), "models/mlp_iris.pth")

    # --- Wine ---
    train_ds, test_ds, class_names = prepare_sklearn_dataset(load_wine)
    model = SimpleMLP(input_dim=13, hidden_dim=32, output_dim=3)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    model = train_model(model, train_loader, test_loader, epochs=30, lr=0.05, device=device)

    acc_train, cm_train = evaluate_model(model, train_loader, device)
    acc_test, cm_test = evaluate_model(model, test_loader, device)
    print("Wine train accuracy:", acc_train)
    print("Wine train confusion matrix:\n", cm_train)
    print("Wine test accuracy:", acc_test)
    print("Wine test confusion matrix:\n", cm_test)
    torch.save(model.state_dict(), "models/mlp_wine.pth")

    # --- Breast Cancer ---
    train_ds, test_ds, class_names = prepare_sklearn_dataset(load_breast_cancer)
    model = SimpleMLP(input_dim=30, hidden_dim=32, output_dim=2)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    model = train_model(model, train_loader, test_loader, epochs=30, lr=0.05, device=device)

    acc_train, cm_train = evaluate_model(model, train_loader, device)
    acc_test, cm_test = evaluate_model(model, test_loader, device)
    print("Breast Cancer train accuracy:", acc_train)
    print("Breast Cancer train confusion matrix:\n", cm_train)
    print("Breast Cancer test accuracy:", acc_test)
    print("Breast Cancer test confusion matrix:\n", cm_test)
    torch.save(model.state_dict(), "models/mlp_breast_cancer.pth")

    # --- MNIST: wszystkie piksele (784D) ---
    train_ds, test_ds = prepare_mnist(flatten_features)
    model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)
    model = train_model(model, train_loader, test_loader, epochs=20, lr=0.1, device=device)

    acc_train, cm_train = evaluate_model(model, train_loader, device)
    acc_test, cm_test = evaluate_model(model, test_loader, device)
    print("MNIST (784D) train accuracy:", acc_train)
    print("MNIST (784D) train confusion matrix:\n", cm_train)
    print("MNIST (784D) test accuracy:", acc_test)
    print("MNIST (784D) test confusion matrix:\n", cm_test)
    torch.save(model.state_dict(), "models/mlp_mnist_784.pth")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)

    # --- MNIST: 4 cechy (ćwiartki) - eksperymenty z parametrami ---
    train_ds, test_ds = prepare_mnist(four_quadrants)
    hidden_sizes = [8, 16, 32, 64]
    lrs = [0.01, 0.05, 0.1]
    batch_sizes = [32, 64, 128]
    epochs_list = [10, 20, 30, 50]
    best_acc = 0
    best_params = None
    best_model = None

    print("Testowanie różnych architektur dla MNIST (quadrants):")
    for hidden_dim in hidden_sizes:
        for lr in lrs:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    model = SimpleMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=10)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_ds, batch_size=batch_size)
                    model = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
                    acc, cm = evaluate_model(model, test_loader, device)
                    print(f"hidden={hidden_dim}, lr={lr}, batch={batch_size}, epochs={epochs}, acc={acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (hidden_dim, lr, batch_size, epochs)
                        best_model = model
    print("Najlepsze parametry (quadrants):", best_params, "accuracy:", best_acc)

    # Ewaluacja na zbiorze treningowym i testowym dla najlepszego modelu (quadrants)
    train_loader = DataLoader(train_ds, batch_size=best_params[2], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=best_params[2], shuffle=False)
    acc_train, cm_train = evaluate_model(best_model, train_loader, device)
    acc_test, cm_test = evaluate_model(best_model, test_loader, device)
    print("Quadrants train accuracy:", acc_train)
    print("Quadrants train confusion matrix:\n", cm_train)
    print("Quadrants test accuracy:", acc_test)
    print("Quadrants test confusion matrix:\n", cm_test)
    torch.save(best_model.state_dict(), "models/mlp_mnist_4d_quad.pth")

    # --- MNIST: PCA (2 komponenty) - eksperymenty z parametrami ---
    n_components = 2
    transform = transforms.ToTensor()
    mnist_train = MNIST(root='.', train=True, download=True, transform=transform)
    X_train = []
    for img, _ in mnist_train:
        img_np = np.array(img, dtype=np.float32).flatten()
        X_train.append(img_np)
    X_train = np.stack(X_train)
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    def pca_features(img, pca=pca):
        flat = img.flatten().reshape(1, -1)
        return pca.transform(flat).flatten()
    train_ds, test_ds = prepare_mnist(pca_features, pca=pca)

    hidden_sizes = [4, 8, 16, 32]
    lrs = [0.01, 0.05, 0.1]
    batch_sizes = [32, 64, 128]
    epochs_list = [10, 20, 30, 50]
    best_acc = 0
    best_params = None
    best_model = None

    print("Testowanie różnych architektur dla MNIST (PCA):")
    for hidden_dim in hidden_sizes:
        for lr in lrs:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    model = SimpleMLP(input_dim=n_components, hidden_dim=hidden_dim, output_dim=10)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_ds, batch_size=batch_size)
                    model = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
                    acc, cm = evaluate_model(model, test_loader, device)
                    print(f"hidden={hidden_dim}, lr={lr}, batch={batch_size}, epochs={epochs}, acc={acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (hidden_dim, lr, batch_size, epochs)
                        best_model = model
    print("Najlepsze parametry (PCA):", best_params, "accuracy:", best_acc)

    # Ewaluacja na zbiorze treningowym i testowym dla najlepszego modelu (PCA)
    train_loader = DataLoader(train_ds, batch_size=best_params[2], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=best_params[2], shuffle=False)
    acc_train, cm_train = evaluate_model(best_model, train_loader, device)
    acc_test, cm_test = evaluate_model(best_model, test_loader, device)
    print("PCA train accuracy:", acc_train)
    print("PCA train confusion matrix:\n", cm_train)
    print("PCA test accuracy:", acc_test)
    print("PCA test confusion matrix:\n", cm_test)
    torch.save(best_model.state_dict(), f"models/mlp_mnist_pca_{n_components}d.pth")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# pozwala na szuaknie najlepszych parametrow dla kazdego modelu.
def hyperparam_search(train_ds, test_ds, input_dim, output_dim,
                     hidden_sizes, lrs, batch_sizes, epochs_list, device, model_name):
    best_acc = 0
    best_params = None
    best_model = None
    for hidden_dim in hidden_sizes:
        for lr in lrs:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_ds, batch_size=batch_size)
                    model = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
                    acc, cm = evaluate_model(model, test_loader, device)
                    print(f"{model_name}: hidden={hidden_dim}, lr={lr}, batch={batch_size}, epochs={epochs}, acc={acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (hidden_dim, lr, batch_size, epochs)
                        best_model = model
    print(f"Najlepsze parametry ({model_name}):", best_params, "accuracy:", best_acc)
    torch.save(best_model.state_dict(), f"models/{model_name}.pth")
    return best_model, best_params, best_acc

# MNIST: 4 cechy (quadrants)
train_ds, test_ds = prepare_mnist(four_quadrants)
hyperparam_search(
    train_ds, test_ds, input_dim=4, output_dim=10,
    hidden_sizes=[8, 16, 32, 64], lrs=[0.01, 0.05, 0.1],
    batch_sizes=[32, 64, 128], epochs_list=[10, 20, 30, 50],
    device=device, model_name="mlp_mnist_4d_quad"
)