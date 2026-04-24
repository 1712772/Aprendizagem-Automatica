

import pandas as pd
import joblib

test_data = pd.read_csv('optdigits.tes', header=None)
clf = joblib.load('modelo_knn.pkl')

# Prever o primeiro dígito do teste
exemplo = test_data.iloc[0:1, :64]
previsao = clf.predict(exemplo)

print(f"O modelo previu o dígito: {previsao[0]}")
print(f"O dígito real era: {test_data.iloc[0, 64]}")