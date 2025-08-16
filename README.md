# Handy: Hand Tracking e Classificação de Gestos em Libras

## 📌 Visão Geral


Este projeto utiliza visão computacional e machine learning para reconhecer gestos da Língua Brasileira de Sinais (Libras) correspondentes ao alfabeto manual. O sistema oferece três modos de operação: treinamento do modelo, reconhecimento de letras e formação de palavras/frases.

## 🛠️ Pré-requisitos

- Python 3.12.6
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- scikit-learn (`pip install scikit-learn`)
- joblib (`pip install joblib`)

Ou, simplesmente, instale as dependências através de `pip install -r requirements.txt`.

## 📂 Estrutura de Arquivos

```
├── hand_landmarker/        # Pasta com o modelo do MediaPipe
│   └── hand_landmarker.task
├── keypoint_test/
│   ├── model/
│   │   ├── keypoint_classifier/
│   │   │   ├── __init__.py
│   │   │   ├── keypoint_classifier_label.csv
│   │   │   ├── keypoint_classifier.keras
│   │   │   ├── keypoint_classifier.hdf5
│   │   │   ├── keypoint_classifier.py
│   │   │   ├── keypoint_classifier.tflite
│   │   │   └── keypoint.csv
│   │   ├── point_history_classifier/
│   │   │   ├── __init__.py
│   │   │   ├── point_history_classifier_label.csv
│   │   │   ├── point_history_classifier.hdf5
│   │   │   ├── point_history_classifier.py
│   │   │   ├── point_history_classifier.tflite
│   │   │   └── point_history.csv
│   │   └── __init__.py
│   ├── __init__.py
│   └── keypoint_classification.py          # Script para treinar o modelo
├── .gitignore     
├── main.py                 # Script principal
├── README.md     
└──  requirements.txt        # Dependências utilizadas
```

## 🎮 Como Usar

### 🔑 Teclas de Controle

| Tecla | Ação |
|-------|------|
| `0` | Ativa modo Treinamento |
| `1` | Ativa modo Normal |
| `2` | Ativa modo Escrita |
| `3` | Ativa modo Escrita (Stop) - pausa captura |
| `Espaço` | Finaliza palavra atual (modo Escrita) |
| `Enter` | Finaliza frase (modo Escrita) |
| `ESC` | Encerra o programa |

### 🧠 Modos de Operação

1. **Modo Normal (Tecla `1`)**
   - Exibe a letra detectada em tempo real
   - Não armazena as letras para formação de palavras

2. **Modo Treinamento (Tecla `0`)**
   - Permite coletar dados para treinar o modelo
   - Pressione uma tecla (a-z) para salvar a posição da mão correspondente
   - Os dados são armazenados em arquivos CSV na pasta `dados_libras`

3. **Modo Escrita (Tecla `2`)**
   - Captura letras para formar palavras e frases
   - Espaço: finaliza a palavra atual
   - Enter: finaliza a frase (exibe no terminal)
   - Cooldown de 0.5s entre letras para evitar duplicações

4. **Modo Escrita Stop (Tecla `3`)**
   - Pausa a captura de letras
   - Mantém o progresso atual da frase
   - Ideal para ajustes ou pausas durante a escrita

## 🚀 Executando o Projeto

1. Clone o repositório
```bash
git clone https://github.com/fcsouzajr/handy.git
```
2. Instale as dependências
```bash
pip install -r requirements.txt
```
3. Execute o script principal:

```bash
python main.py
```

## 🖥️ Saída do Programa

- Janela com visualização da câmera e landmarks da mão
- Modo atual exibido no topo da tela
- Letra detectada exibida na tela
- No modo Escrita: exibe palavra e frase em formação
- No terminal: feedback das ações realizadas

## 🤖 Treinando o Modelo

1. Entre no modo Treinamento (tecla `0`)
2. Para cada letra do alfabeto:
   - Posicione a mão no gesto correspondente
   - Pressione a tecla da letra (a-z) para capturar
   - Repita várias vezes para criar um conjunto de dados robusto
3. Após coletar dados para várias letras, treine o modelo com:
```python
 cd keypoint_test
 python keypoint_classification.py
```

## 💡 Dicas

### Durante o uso
- Mantenha a mão bem visível para a câmera
- No modo Escrita, mantenha o gesto por pelo menos 0.5s para ser capturado
- Use o modo Escrita Stop quando precisar fazer pausas
- Para melhor reconhecimento, treine o modelo com várias amostras de cada letra

### Durante o treinamento
- Capture pelo menos 50-100 amostras por letra
- Evite deixar as quantidades de dados de cada letra muito diferentes, para evitar classificações enviesadas
- Mantenha iluminação adequada durante a captura
- Execute `keypoint_classification.py` para gerar a nova versão do modelo

## 🖐️ Sobre o Hand Landmarker
Utilizou-se de base para fazer o handtracking o modelo "Hand Landmarker" disponibilizado pelo MediaPipe do [Google AI for Developers](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=pt-br). Ele permite detectar 21 pontos de referência (landmarks) das mãos em uma imagem, como ilustrado abaixo: 

![Pontos de referência (landmarks) da mão detectados pelo MediaPipe](https://ai.google.dev/static/edge/mediapipe/images/solutions/hand-landmarks.png)

*Figura 1: Os 21 landmarks da mão identificados pelo modelo Hand Landmarker*

Além de identificar com precisão 21 pontos anatômicos da mão, possui suporte para detecção simultânea de múltiplas mãos.
