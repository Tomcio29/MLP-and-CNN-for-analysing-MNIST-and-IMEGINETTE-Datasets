import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Pomocnicze funkcje do wizualizacji ---

def plot_decision_boundary(clf, X, y, title, ax):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict", alpha=0.4, ax=ax, cmap=plt.cm.coolwarm
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    ax.set_title(title)


def plot_confusion(y_true, y_pred, title, ax, cm_font_size=25):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    for text in disp.text_.ravel():
        text.set_fontsize(cm_font_size)

# --- Funkcja do ładowania danych ---

def load_data(filename):
    data = pd.read_csv(filename, header=None, delimiter=";")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# --- Eksperyment 1: Elastyczność granic decyzyjnych SVM i MLP ---

def experiment1(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # SVM: kernel linear i rbf
    for kernel in ["linear", "rbf"]:
        best_acc = 0
        best_C = None
        for C in [0.01, 0.1, 1, 10, 100]:
            clf = SVC(kernel=kernel, C=C, random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_train, clf.predict(X_train))
            if acc > best_acc:
                best_acc = acc
                best_C = C
        clf = SVC(kernel=kernel, C=best_C, random_state=42)
        clf.fit(X_train, y_train)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_decision_boundary(clf, X_train, y_train, f"{dataset_name} | SVM-{kernel} train", axs[0])
        plot_decision_boundary(clf, X_test, y_test, f"{dataset_name} | SVM-{kernel} test", axs[1])
        plt.suptitle(f"{dataset_name} | SVM kernel={kernel}, C={best_C}")
        plt.tight_layout()
        plt.show()

    # MLP: activation identity i relu
    for activation in ["identity", "relu"]:
        best_acc = 0
        best_n = None
        for n in [2, 4, 8, 16, 32, 64]:
            clf = MLPClassifier(hidden_layer_sizes=(n,), activation=activation,
                               max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_train, clf.predict(X_train))
            if acc > best_acc:
                best_acc = acc
                best_n = n
        clf = MLPClassifier(hidden_layer_sizes=(best_n,), activation=activation,
                            max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)
        clf.fit(X_train, y_train)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_decision_boundary(clf, X_train, y_train, f"{dataset_name} | MLP-{activation} train", axs[0])
        plot_decision_boundary(clf, X_test, y_test, f"{dataset_name} | MLP-{activation} test", axs[1])
        plt.suptitle(f"{dataset_name} | MLP activation={activation}, neurons={best_n}")
        plt.tight_layout()
        plt.show()



# --- Eksperyment 2: K-NN, SVM, MLP - accuracy i granice decyzyjne (duży zbiór treningowy) ---

def experiment2(X, y, dataset_name, method, train_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if method == "knn":
        param_range = range(1, 21)
        train_scores, test_scores = [], []
        for k in param_range:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        plt.plot(param_range, train_scores, label='Train')
        plt.plot(param_range, test_scores, label='Test')
        plt.xlabel('n_neighbors')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_name} | K-NN: accuracy vs n_neighbors')
        plt.legend()
        plt.show()
        best_k = param_range[np.argmax(test_scores)]
        selected = [param_range[0], best_k, param_range[-1]]
        for k in selected:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            plot_decision_boundary(clf, X_train, y_train, f"{dataset_name} | KNN train, k={k}", axs[0, 0])
            plot_confusion(y_train, clf.predict(X_train), f"{dataset_name} | Confusion train, k={k}", axs[1, 0])
            plot_decision_boundary(clf, X_test, y_test, f"{dataset_name} | KNN test, k={k}", axs[0, 1])
            plot_confusion(y_test, clf.predict(X_test), f"{dataset_name} | Confusion test, k={k}", axs[1, 1])
            plt.tight_layout()
            plt.show()

    elif method == "svm":
        Cs = np.logspace(-2, 6, num=32)
        exponents = np.log10(Cs)
        train_scores, test_scores = [], []
        for C in Cs:
            clf = SVC(kernel="rbf", C=C, random_state=42)
            clf.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        plt.plot(exponents, train_scores, label='Train')
        plt.plot(exponents, test_scores, label='Test')
        plt.xlabel('log10(C)')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_name} | SVM: accuracy vs log10(C)')
        major_ticks = np.arange(-2, 7, 1)  # -2, -1, ..., 6
        plt.xticks(major_ticks, labels=[str(e) for e in major_ticks])
        plt.legend()
        plt.show()
        best_C = Cs[np.argmax(test_scores)]
        selected = [Cs[0], best_C, Cs[-1]]
        for C in selected:
            clf = SVC(kernel="rbf", C=C, random_state=42)
            clf.fit(X_train, y_train)
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            plot_decision_boundary(clf, X_train, y_train, f"SVM train, C={C:.2f}", axs[0, 0])
            plot_confusion(y_train, clf.predict(X_train), f"Confusion train, C={C:.2f}", axs[1, 0])
            plot_decision_boundary(clf, X_test, y_test, f"SVM test, C={C:.2f}", axs[0, 1])
            plot_confusion(y_test, clf.predict(X_test), f"Confusion test, C={C:.2f}", axs[1, 1])
            plt.tight_layout()
            plt.show()

    elif method == "mlp":
        param_range = np.linspace(1, 64, num=16, dtype=int)
        train_scores, test_scores = [], []
        for n in param_range:
            clf = MLPClassifier(hidden_layer_sizes=(n,), activation='relu',
                               max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)
            clf.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        plt.plot(param_range, train_scores, label='Train')
        plt.plot(param_range, test_scores, label='Test')
        plt.xlabel('neurons')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_name} | MLP: accuracy vs hidden_layer_size')
        plt.legend()
        plt.show()
        best_n = param_range[np.argmax(test_scores)]
        selected = [param_range[0], best_n, param_range[-1]]
        for n in selected:
            clf = MLPClassifier(hidden_layer_sizes=(n,), activation='relu',
                               max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd', random_state=42)
            clf.fit(X_train, y_train)
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            plot_decision_boundary(clf, X_train, y_train, f"{dataset_name} | MLP train, n={n}", axs[0, 0])
            plot_confusion(y_train, clf.predict(X_train), f"{dataset_name} | Confusion train, n={n}", axs[1, 0])
            plot_decision_boundary(clf, X_test, y_test, f"{dataset_name} | MLP test, n={n}", axs[0, 1])
            plot_confusion(y_test, clf.predict(X_test), f"{dataset_name} | Confusion test, n={n}", axs[1, 1])
            plt.tight_layout()
            plt.show()

# --- Eksperyment 4: Nauka MLP po epokach i wielokrotne uruchomienia ---

def plot_decision_boundary_epoch(clf, X_train, y_train, X_test, y_test, train_acc, test_acc, dataset_name, epoch_desc):
    """
    Rysuje granicę decyzyjną dla modelu na zbiorach treningowym i testowym.
    Tytuły wykresów zawierają tylko odpowiednie accuracy.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Granica decyzyjna na danych treningowych
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, response_method="predict", alpha=0.4, ax=axs[0], cmap=plt.cm.coolwarm
    )
    axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm)
    axs[0].set_title(f'{dataset_name} | Train acc: {train_acc:.3f}\n{epoch_desc}')

    # Granica decyzyjna na danych testowych
    DecisionBoundaryDisplay.from_estimator(
        clf, X_test, response_method="predict", alpha=0.4, ax=axs[1], cmap=plt.cm.coolwarm
    )
    axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap=plt.cm.coolwarm)
    axs[1].set_title(f'{dataset_name} | Test acc: {test_acc:.3f}\n{epoch_desc}')

    plt.tight_layout()
    return fig


def experiment4(X, y, dataset_name, neurons, train_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_epochs = 100000
    train_scores = []
    test_scores = []

    # Tworzymy klasyfikator i trenujemy go epoka po epoce
    clf = MLPClassifier(hidden_layer_sizes=(neurons,), activation='relu',
                        max_iter=1, warm_start=True, solver='sgd', random_state=42)

    # Przechowujemy modele dla wybranych epok
    first_epoch_model = None
    best_epoch_model = None
    best_epoch = 0
    best_test_score = 0

    for epoch in range(n_epochs):
        clf.fit(X_train, y_train)
        train_score = accuracy_score(y_train, clf.predict(X_train))
        test_score = accuracy_score(y_test, clf.predict(X_test))

        train_scores.append(train_score)
        test_scores.append(test_score)

        # Zapisujemy model pierwszej epoki
        if epoch == 0:
            # Trenujemy nowy model tylko dla pierwszej epoki
            first_epoch_model = MLPClassifier(hidden_layer_sizes=(neurons,), activation='relu',
                                              max_iter=1, solver='sgd', random_state=42)
            first_epoch_model.fit(X_train, y_train)

        # Aktualizujemy najlepszy model
        if test_score > best_test_score:
            best_test_score = test_score
            best_epoch = epoch + 1  # epoki numerujemy od 1
            # Trenujemy nowy model dla najlepszej epoki
            best_epoch_model = MLPClassifier(hidden_layer_sizes=(neurons,), activation='relu',
                                             max_iter=best_epoch, solver='sgd', random_state=42)
            best_epoch_model.fit(X_train, y_train)

    # Model dla ostatniej epoki to aktualny model
    last_epoch_model = MLPClassifier(hidden_layer_sizes=(neurons,), activation='relu',
                                     max_iter=n_epochs, solver='sgd', random_state=42)
    last_epoch_model.fit(X_train, y_train)

    # Rysujemy krzywą uczenia
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), train_scores, label='Train')
    plt.plot(range(1, n_epochs + 1), test_scores, label='Test')
    plt.axvline(x=best_epoch, color='r', linestyle='--',
                label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} | MLP learning curve (neurons={neurons}, train_size={train_size})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Rysujemy granice decyzyjne dla trzech epok
    plot_decision_boundary_epoch(
        first_epoch_model, X_train, y_train, X_test, y_test,
        train_acc=train_scores[0],
        test_acc=test_scores[0],
        dataset_name=dataset_name,
        epoch_desc="First Epoch"
    )
    plt.show()

    # Przykład dla najlepszej epoki
    plot_decision_boundary_epoch(
        best_epoch_model, X_train, y_train, X_test, y_test,
        train_acc=train_scores[best_epoch - 1],
        test_acc=test_scores[best_epoch - 1],
        dataset_name=dataset_name,
        epoch_desc=f"Best Epoch {best_epoch}"
    )
    plt.show()

    # Przykład dla ostatniej epoki
    plot_decision_boundary_epoch(
        last_epoch_model, X_train, y_train, X_test, y_test,
        train_acc=train_scores[-1],
        test_acc=test_scores[-1],
        dataset_name=dataset_name,
        epoch_desc=f"Last Epoch {n_epochs}"
    )
    plt.show()

    # Wielokrotne uruchomienia (10 razy)
    results = []
    for seed in range(10):
        # Trenujemy model epoka po epoce
        clf = MLPClassifier(hidden_layer_sizes=(neurons,), activation='relu',
                            max_iter=1, warm_start=True, solver='sgd', random_state=seed)

        run_train_scores = []
        run_test_scores = []

        for epoch in range(n_epochs):
            clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            run_train_scores.append(train_acc)
            run_test_scores.append(test_acc)

        # Znajdujemy najlepszą epokę (maksymalne test_acc)
        best_run_epoch_idx = np.argmax(run_test_scores)
        best_run_epoch = best_run_epoch_idx + 1  # epoki numerujemy od 1

        # Zapisujemy wyniki dla pierwszej, najlepszej i ostatniej epoki
        result = {
            'Run': seed + 1,
            'Train Acc First': run_train_scores[0],
            'Test Acc First': run_test_scores[0],
            'Train Acc Best': run_train_scores[best_run_epoch_idx],
            'Test Acc Best': run_test_scores[best_run_epoch_idx],
            'Best Epoch': best_run_epoch,
            'Train Acc Last': run_train_scores[-1],
            'Test Acc Last': run_test_scores[-1]
        }
        results.append(result)

    # Tworzymy dataframe z wynikami i wyświetlamy go jako tabelę
    results_df = pd.DataFrame(results)
    print(f"\nResults for {dataset_name} (neurons={neurons}, train_size={train_size}):")
    print(results_df.to_string(index=False, float_format='%.3f'))

    return results_df


# --- Przykład użycia ---

if __name__ == "__main__":
    # for fname in ["2_1.csv", "2_2.csv", "2_3.csv"]:
    #     X, y = load_data(fname)
    #     dataset_name = fname.split(".")[0]
    #     experiment1(X, y, dataset_name)

    for fname in ["2_2.csv", "2_3.csv"]:
        X, y = load_data(fname)
        dataset_name = fname.split(".")[0]
        # print(f"== Eksperyment 2 dla {fname} ==")
        # experiment2(X, y, dataset_name, "knn", train_size=0.8)
        # experiment2(X, y, dataset_name, "svm", train_size=0.8)
        # experiment2(X, y, dataset_name, "mlp", train_size=0.8)
        # print(f"== Eksperyment 3 dla {fname} ==")
        # experiment2(X, y, dataset_name, "knn", train_size=0.2)
        # experiment2(X, y, dataset_name, "svm", train_size=0.2)
        # experiment2(X, y, dataset_name, "mlp", train_size=0.2)
    for fname in ["2_3.csv"]:
        X, y = load_data(fname)
        dataset_name = fname.split(".")[0]
        print(f"== Eksperyment 4 dla {fname} ==")
        experiment4(X, y, dataset_name, neurons=9, train_size=0.8)
        experiment4(X, y, dataset_name, neurons=55, train_size=0.2)
