import pandas as pd

# Lê ignorando linhas com erro
df = pd.read_csv("keypoint.csv", header=None, on_bad_lines="skip")

# Filtra onde a primeira coluna não é 21
df_filtrado = df[df[0] != 21]

# Salva de volta
df_filtrado.to_csv("dados_filtrados.csv", index=False, header=False)