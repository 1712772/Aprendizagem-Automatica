import pandas as pd
import joblib

# Atividade 1: Carregar dados de teste
test_data = pd.read_csv('optdigits.tes', header=None)
X_test = test_data.iloc[:, :64]
y_test = test_data.iloc[:, 64]

# CARREGAR o modelo treinado (evita o erro de 'clf' não definido)
clf = joblib.load('modelo_knn.pkl')

# Atividade 3: Avaliação da precisão
precisao = clf.score(X_test, y_test)
print(f"Atividade 3 - Avaliação:")
print(f"A precisão do modelo no conjunto de teste é: {precisao * 100:.2f}%")