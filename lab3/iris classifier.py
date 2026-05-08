# =====================================
# LABORATÓRIO 3
# DIGITS + PCA + KNN
# =====================================

# ============
# BIBLIOTECAS
# ============

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =================
# CARREGAR DATASET
# =================

digits = load_digits()

X = digits.data
y = digits.target

print("Dimensão original:", X.shape)

# =========================
# DIVISÃO TREINO E TESTE
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# =========================
# REDUÇÃO DE DIMENSÃO PCA
# =========================

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Dimensão reduzida:", X_train_pca.shape)

# =========================
# TREINO DO MODELO
# =========================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_pca, y_train)

print("Modelo treinado!")

# =========================
# TESTE DO MODELO
# =========================

score = knn.score(X_test_pca, y_test)

print("Precisão do modelo:", score)

# =========================
# GUARDAR MODELO
# =========================

with open("modelo_knn.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Modelo guardado!")

# =========================
# PREDIÇÃO
# =========================

indice = 15

imagem = X_test[indice]

imagem_pca = pca.transform([imagem])

predicao = knn.predict(imagem_pca)

print("Dígito previsto:", predicao[0])

# =========================
# MOSTRAR IMAGEM
# =========================

plt.figure(figsize=(4, 4))

plt.imshow(imagem.reshape(8, 8), cmap="gray")

plt.title(f"Predição: {predicao[0]}")

plt.axis("off")

plt.show()

# =========================
# VISUALIZAÇÃO GRÁFICA
# =========================

y_pred = knn.predict(X_test_pca)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_test_pca[:, 0],
    y=X_test_pca[:, 1],
    hue=y_pred,
    palette="tab10"
)

plt.title("Classificação dos Dígitos com PCA + KNN")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()