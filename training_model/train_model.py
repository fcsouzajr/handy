import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Caminho dos dados
dados_dir = "dados_libras"

X = []
y = []

# Lê todos os arquivos CSV e extrai os dados
for arquivo in os.listdir(dados_dir):
    if arquivo.endswith(".csv"):
        label = arquivo[0]  # Assume que o nome é "a.csv", "b.csv"...
        caminho = os.path.join(dados_dir, arquivo)
        with open(caminho, 'r') as f:
            leitor = csv.reader(f)
            for linha in leitor:
                if len(linha) == 64:  # 21 pontos * 3 (x,y,z) + 1 label
                    X.append([float(i) for i in linha[:-1]])
                    y.append(label)

# Transforma em arrays
X = np.array(X)
y = np.array(y)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treina modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salva modelo
joblib.dump(modelo, "modelo_libras.pkl")
print("Modelo salvo como 'modelo_libras.pkl'")