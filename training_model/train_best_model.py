import os
import csv
import numpy as np
import joblib
from time import time
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Modelos a serem testados
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Carregar dados
def carregar_dados(dados_dir="../dados_libras"):
    X, y = [], []
    for arquivo in os.listdir(dados_dir):
        if arquivo.endswith(".csv"):
            label = arquivo[0]
            caminho = os.path.join(dados_dir, arquivo)
            with open(caminho, 'r') as f:
                leitor = csv.reader(f)
                for linha in leitor:
                    if len(linha) == 64:  # 21 pontos * 3 + label
                        X.append([float(i) for i in linha[:-1]])
                        y.append(label)
    return np.array(X), np.array(y)

# Definir todos os modelos
def inicializar_modelos():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100),
        'Bagging': BaggingClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(n_estimators=100),
        'CatBoost': CatBoostClassifier(iterations=100, verbose=0)
    }

# Função para testar modelos
def testar_modelos(X_train, X_test, y_train, y_test, modelos):
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"\n=== Treinando {nome} ===")
        start_time = time()
        
        try:
            modelo.fit(X_train, y_train)
            train_time = time() - start_time
            
            start_pred = time()
            y_pred = modelo.predict(X_test)
            pred_time = time() - start_pred
            
            resultados[nome] = {
                'model': modelo,
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'train_time': train_time,
                'pred_time': pred_time
            }
            
            print(f"\n{nome}:")
            print(f"Acurácia: {resultados[nome]['accuracy']:.4f}")
            print(f"Tempo Treino: {train_time:.2f}s")
            print(f"Tempo Predição: {pred_time:.4f}s/amostra")
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            print(f"Erro no modelo {nome}: {str(e)}")
            resultados[nome] = None
    
    return resultados

# Função principal
def main():
    # Carregar dados
    X, y = carregar_dados()
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Inicializar modelos
    modelos = inicializar_modelos()
    
    # Testar modelos
    resultados = testar_modelos(X_train, X_test, y_train, y_test, modelos)
    
    # Encontrar e salvar o melhor modelo
    modelos_validos = {k: v for k, v in resultados.items() if v is not None}
    if modelos_validos:
        melhor_nome = max(modelos_validos, key=lambda x: modelos_validos[x]['accuracy'])
        melhor_modelo = modelos_validos[melhor_nome]['model']
        
        print(f"\n=== Melhor modelo: {melhor_nome} ===")
        print(f"Acurácia: {modelos_validos[melhor_nome]['accuracy']:.4f}")
        
        joblib.dump(melhor_modelo, "modelo_libras.pkl")
        print("Melhor modelo salvo como 'modelo_libras.pkl'")
        
        # Salvar relatório completo
        with open("relatorio_modelos.txt", "w") as f:
            for nome, res in modelos_validos.items():
                f.write(f"\n=== {nome} ===\n")
                f.write(f"Acurácia: {res['accuracy']:.4f}\n")
                f.write(f"Tempo Treino: {res['train_time']:.2f}s\n")
                f.write(f"Tempo Predição: {res['pred_time']:.4f}s/amostra\n")
                
                # Corrigindo a escrita do relatório
                report = res['report']
                if isinstance(report, dict):
                    # Formatar o relatório de classificação manualmente
                    f.write("\nClassification Report:\n")
                    f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}\n")
                    for class_name, metrics in report.items():
                        if isinstance(metrics, dict):  # Linhas por classe
                            f.write(f"{class_name:<10} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {metrics['support']:<10}\n")
                    # Adicionar métricas globais
                    f.write(f"\nAccuracy: {report['accuracy']:.2f}\n")
                    f.write(f"Macro avg: precision={report['macro avg']['precision']:.2f}, recall={report['macro avg']['recall']:.2f}, f1-score={report['macro avg']['f1-score']:.2f}\n")
                    f.write(f"Weighted avg: precision={report['weighted avg']['precision']:.2f}, recall={report['weighted avg']['recall']:.2f}, f1-score={report['weighted avg']['f1-score']:.2f}\n")
                else:
                    f.write("\nClassification Report:\n")
                    f.write(report)
    else:
        print("Nenhum modelo foi treinado com sucesso.")

if __name__ == "__main__":
    main()