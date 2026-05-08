# =====================================
# LABORATÓRIO 3
# DIGITS + PCA + KNN
# =====================================

# ============
# BIBLIOTECAS
# ============

import numpy as np
import matplotlib.pyplot as plt
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

print("Dados divididos!")

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

print("Modelo KNN guardado!")

# =========================
# GUARDAR PCA
# =========================

with open("modelo_pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("Modelo PCA guardado!")

# =========================
# GUARDAR DADOS
# =========================

dados = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "X_train_pca": X_train_pca,
    "X_test_pca": X_test_pca
}

with open("dados.pkl", "wb") as f:
    pickle.dump(dados, f)

print("Dados guardados!")

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
# CARREGAR MODELOS
# =========================

with open("modelo_knn.pkl", "rb") as f:
    knn_carregado = pickle.load(f)

with open("modelo_pca.pkl", "rb") as f:
    pca_carregado = pickle.load(f)

with open("dados.pkl", "rb") as f:
    dados_carregados = pickle.load(f)

print("Tudo carregado com sucesso!")

# =========================
# NOVO TESTE
# =========================

nova_imagem = dados_carregados["X_test"][20]

nova_imagem_pca = pca_carregado.transform([nova_imagem])

nova_predicao = knn_carregado.predict(nova_imagem_pca)

print("Nova previsão:", nova_predicao[0])

# =========================
# MOSTRAR NOVA IMAGEM
# =========================

plt.figure(figsize=(4, 4))

plt.imshow(nova_imagem.reshape(8, 8), cmap="gray")

plt.title(f"Nova previsão: {nova_predicao[0]}")

plt.axis("off")

plt.show()
import seaborn as sns

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