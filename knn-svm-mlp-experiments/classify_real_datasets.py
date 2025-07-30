import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# --- Funkcja do ładowania zbiorów rzeczywistych ---
def load_real_dataset(name):
    if name == "wine":
        data = load_wine()
    elif name == "iris":
        data = load_iris()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError("Unknown dataset")
    X, y = data.data, data.target
    return X, y, data.target_names

# --- Funkcja do eksperymentów na zbiorach rzeczywistych ---
def real_data_experiments(dataset_name):
    print(f"\n=== {dataset_name.upper()} ===")
    X, y, target_names = load_real_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- KNN ---
    param_range_knn = range(1, 21)
    train_scores_knn, test_scores_knn = [], []
    for k in param_range_knn:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        train_scores_knn.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores_knn.append(accuracy_score(y_test, clf.predict(X_test)))
    best_k = param_range_knn[np.argmax(test_scores_knn)]
    clf_knn = KNeighborsClassifier(n_neighbors=best_k)
    clf_knn.fit(X_train, y_train)
    y_pred_knn = clf_knn.predict(X_test)

    # Wykres accuracy vs k
    plt.figure()
    plt.plot(param_range_knn, train_scores_knn, label='Train')
    plt.plot(param_range_knn, test_scores_knn, label='Test')
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} | KNN: accuracy vs n_neighbors')
    plt.legend()
    plt.show()

    # --- SVM ---
    Cs = np.logspace(-2, 6, num=32)
    exponents = np.log10(Cs)
    train_scores_svm, test_scores_svm = [], []
    for C in Cs:
        clf = SVC(kernel="rbf", C=C, random_state=42)
        clf.fit(X_train, y_train)
        train_scores_svm.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores_svm.append(accuracy_score(y_test, clf.predict(X_test)))
    best_C = Cs[np.argmax(test_scores_svm)]
    clf_svm = SVC(kernel="rbf", C=best_C, random_state=42)
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)

    # Wykres accuracy vs log10(C)
    plt.figure()
    plt.plot(exponents, train_scores_svm, label='Train')
    plt.plot(exponents, test_scores_svm, label='Test')
    plt.xlabel('log10(C)')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} | SVM: accuracy vs log10(C)')
    plt.xticks(np.arange(-2, 7, 1))
    plt.legend()
    plt.show()

    # --- MLP ---
    param_range_mlp = np.linspace(1, 64, num=16, dtype=int)
    train_scores_mlp, test_scores_mlp = [], []
    for n in param_range_mlp:
        clf = MLPClassifier(hidden_layer_sizes=(n,), activation='relu',
                            max_iter=10000, tol=1e-4, solver='sgd', random_state=42)
        clf.fit(X_train, y_train)
        train_scores_mlp.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores_mlp.append(accuracy_score(y_test, clf.predict(X_test)))
    best_n = param_range_mlp[np.argmax(test_scores_mlp)]
    clf_mlp = MLPClassifier(hidden_layer_sizes=(best_n,), activation='relu',
                           max_iter=10000, tol=1e-4, solver='sgd', random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred_mlp = clf_mlp.predict(X_test)

    # Wykres accuracy vs neurons
    plt.figure()
    plt.plot(param_range_mlp, train_scores_mlp, label='Train')
    plt.plot(param_range_mlp, test_scores_mlp, label='Test')
    plt.xlabel('neurons')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} | MLP: accuracy vs hidden_layer_size')
    plt.legend()
    plt.show()

    # --- Wyświetlenie wyników w tabelce ---
    # KNN: najmniejszy, najlepszy, największy k
    knn_params = [param_range_knn[0], best_k, param_range_knn[-1]]
    knn_acc = []
    for k in knn_params:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))
        knn_acc.append((k, acc_train, acc_test))
    df_knn = pd.DataFrame(knn_acc, columns=['k', 'Train Accuracy', 'Test Accuracy'])
    print("\nKNN accuracy for selected k values:")
    print(df_knn.to_string(index=False))

    # SVM: najmniejszy, najlepszy, największy C
    svm_params = [Cs[0], best_C, Cs[-1]]
    svm_acc = []
    for C in svm_params:
        clf = SVC(kernel="rbf", C=C, random_state=42)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))
        svm_acc.append((C, acc_train, acc_test))
    df_svm = pd.DataFrame(svm_acc, columns=['C', 'Train Accuracy', 'Test Accuracy'])
    print("\nSVM accuracy for selected C values:")
    print(df_svm.to_string(index=False))

    # MLP: najmniejszy, najlepszy, największy n
    mlp_params = [param_range_mlp[0], best_n, param_range_mlp[-1]]
    mlp_acc = []
    for n in mlp_params:
        clf = MLPClassifier(hidden_layer_sizes=(n,), activation='relu',
                            max_iter=10000, tol=1e-4, solver='sgd', random_state=42)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))
        mlp_acc.append((n, acc_train, acc_test))
    df_mlp = pd.DataFrame(mlp_acc, columns=['neurons', 'Train Accuracy', 'Test Accuracy'])
    print("\nMLP accuracy for selected neuron counts:")
    print(df_mlp.to_string(index=False))

    # --- Macierze pomyłek ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_estimator(clf_knn, X_test, y_test, ax=axs[0], display_labels=target_names)
    axs[0].set_title(f'KNN (k={best_k})')
    ConfusionMatrixDisplay.from_estimator(clf_svm, X_test, y_test, ax=axs[1], display_labels=target_names)
    axs[1].set_title(f'SVM (C={best_C:.3g})')
    ConfusionMatrixDisplay.from_estimator(clf_mlp, X_test, y_test, ax=axs[2], display_labels=target_names)
    axs[2].set_title(f'MLP (neurons={best_n})')
    plt.suptitle(f"{dataset_name.upper()} - Confusion Matrices")
    plt.tight_layout()
    plt.show()


# --- Uruchomienie eksperymentów dla wszystkich zbiorów ---
for ds in ["wine", "iris", "breast_cancer"]:
    real_data_experiments(ds)
