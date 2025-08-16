import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight 

import pyttsx3

RANDOM_SEED = 42

dataset = 'model/keypoint_classifier/keypoint.csv'

model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'

NUM_CLASSES = 27

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

unique_classes = np.unique(y_train)
weight_classes = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)
weight_classes = dict(enumerate(weight_classes))

# Aumentar peso manualmente para M, N, U, V
weight_classes[12] *= 4.0  # M
weight_classes[13] *= 2.5  # N
weight_classes[20] *= 4.0  # U
weight_classes[21] *= 3.0  # V

def oversample(X, y, target_classes, factor=3):
    X_res, y_res = [X], [y]
    for c in target_classes:
        idx = np.where(y == c)[0]
        X_c = X[idx]
        y_c = y[idx]
        for _ in range(factor - 1):
            X_res.append(X_c)
            y_res.append(y_c)
    return np.vstack(X_res), np.hstack(y_res)

X_train, y_train = oversample(X_train, y_train, target_classes=[19, 20], factor=5)

model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

# モデルチェックポイントのコールバック
tmp_model_path = 'model/keypoint_classifier/keypoint_classifier.keras'

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

# 早期打ち切り用コールバック
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

#model_save_path = "model/keypoint_classifier/keypoint_classifier.keras"


#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    model_save_path, verbose=1, save_weights_only=False)

#es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

print("""===================
PARTE 1 COMPLETA
===================""")


# モデルコンパイル
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback], 
    class_weight=weight_classes
)

print("""===================
PARTE 2 COMPLETA
===================""")

# モデル評価
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

# 保存したモデルのロード
#model = tf.keras.models.load_model(model_save_path)

try:
    model = tf.keras.models.load_model(model_save_path)
except Exception as e:
    print("Erro ao carregar o modelo:", e)
    try:
        model = tf.keras.models.load_model('model/keypoint_classifier/keypoint_classifier')
    except Exception as e:
        print("Erro ao carregar o modelo SavedModel:", e)


print("""===================
PARTE 3 COMPLETA
===================""")

# 推論テスト
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


print("""===================
PARTE 4 COMPLETA
===================""")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

print("""===================
PARTE 5 COMPLETA
===================""")

# 推論専用のモデルとして保存
model.save(model_save_path, include_optimizer=True)

# モデルを変換(量子化)
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

print("""===================
PARTE 6 COMPLETA
===================""")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

print("""===================
PARTE 7 COMPLETA
===================""")

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

# 入出力テンソルを取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# 推論実施
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

print("""===================
COMPLETO
===================""")