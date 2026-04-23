import pandas as pd
from sklearn import neighbors
import joblib

# Carregar dados de treino
train_data = pd.read_csv('optdigits.tra', header=None)
X_train = train_data.iloc[:, :64]
y_train = train_data.iloc[:, 64]

# Treinar o modelo
clf = neighbors.KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)

# ESTA LINHA CRIA O FICHEIRO QUE ESTÁ A FALTAR
joblib.dump(clf, 'modelo_knn.pkl')

print("Sucesso: O ficheiro 'modelo_knn.pkl' foi criado na pasta!")