import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# prelucrarea datelor
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    return train_data, test_data


# grafic distributia claselor în setul de date
def visualize_distribution(data, title):
    plt.figure(figsize=(20, 10))
    class_counts = data.value_counts() # numarul de aparitii pentru fiecare clasa din setul de date 
    colors = ['purple', 'green', 'blue', 'pink', 'orange'] 
    class_counts.plot(kind='bar', color=colors)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Echilibram setul de date prin resamplare, pentru a ne asigura ca fiecare clasa are acelasi numar de exemple
def balance_data(data, num_samples, target_column):
    balanced_data = [resample(data[data[target_column] == i], replace=True, n_samples=num_samples, random_state=42 + int(i)) for i in data[target_column].unique()]
    return pd.concat(balanced_data)

# CNN - convolutional neural network
def build_model(input_dim, num_classes):
    # Strat de intrare
    input_layer = Input(shape=(input_dim, 1), name='input_layer')
    
    # Primul strat de convoluție și max pooling cu 32 de filtre
    layer = Conv1D(32, 6, activation='relu')(input_layer)
    layer = MaxPooling1D(pool_size=3, strides=2, padding="same")(layer)
    
    # Al doilea strat de convoluție și max pooling cu 64 de filtre
    layer = Conv1D(64 , 3, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=3, strides=2, padding="same")(layer)
    
    # Al treilea strat de convoluție și max pooling cu 128 de filtre
    layer = Conv1D(128 , 3, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same")(layer)
    
    # Flatten layer
    layer = Flatten()(layer)
    
    # Straturi dense (fully-connected)
    layer = Dense(128, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    
    # Strat final (output layer)
    final_layer = Dense(num_classes, activation='softmax', name='final_layer')(layer)
    
    model = Model(inputs=input_layer, outputs=final_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Antrenare si evaluare modelul
def train_evaluate(model, train_data, train_labels, validation_data, validation_labels, epochs=10, batch_size=32):
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), callbacks=callbacks)
    model.load_weights('best_model.h5')
    return history

# Prelucrarea datelor
train_path = 'C:/Users/Iulian/Downloads/SIRT/mitbih_train.csv'
test_path = 'C:/Users/Iulian/Downloads/SIRT/mitbih_test.csv'

train_data, test_data = load_data(train_path, test_path)

# distributia setului de date din cele 5 clase
visualize_distribution(train_data[187], 'Class Distribution')

train_data = balance_data(train_data, 20000, 187)
visualize_distribution(train_data[187], 'Class Distribution')



# Preluarea datelor și împărțirea acestora în seturi de antrenament și test
def preprocess_data(train_data, test_data, target_column=187):
    X_train = train_data.iloc[:, :target_column].values
    X_test = test_data.iloc[:, :target_column].values
    # matrice de etichete pentru setul de antrenare  
    y_train = to_categorical(train_data[target_column])
    # matrice de etichete pentru setul de testare  
    y_test = to_categorical(test_data[target_column])
    return X_train, X_test, y_train, y_test

# Redimensionam datele pentru modelul Conv1D
def resize_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test


X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
X_train, X_test = resize_data(X_train, X_test)

# Construirea și antrenarea modelului
num_classes = y_train.shape[1]
input_dim = X_train.shape[1]
model = build_model(input_dim, num_classes)
history = train_evaluate(model, X_train, y_train, X_test, y_test)

# Evaluarea performanței modelului
def evaluate_performance(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {scores[1] * 100:.2f}%")

evaluate_performance(model, X_test, y_test)


# Exemplu de ECG pentru fiecare clasă
def plot_1beat_ecg_examples(X, y):
    plt.figure(figsize=(20,6))
    for i in range(5): 
        idx = np.where(y[:, i] == 1)[0][0] 
        plt.plot(X[idx], label=f"Class {i}")
        plt.legend()
        plt.title(f"Class {i}", fontsize=20)
        plt.ylabel("Amplitude", fontsize=15)
        plt.xlabel("Time Points", fontsize=15)
        plt.show()

#  Exemple de ECG
plot_1beat_ecg_examples(X_train, y_train)

import matplotlib.pyplot as plt

# Acuratețe pe setul de antrenament și validare
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')

# Pierdere pe setul de antrenament și validare
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')

plt.show()


# Funcția pentru a afișa curba ROC și calcularea AUC
def plot_roc_curve(y_test, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Matricea de confuzie
def plot_confusion_matrix(y_true_classes, y_pred_classes, num_classes, figsize=(10, 8), class_labels=None):
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    # Afisam matricea de confuzie sub forma unui cadru de date
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.colorbar(ax.collections[0])
    plt.show()

# y_pred reprezinta probabilitatile prezise de model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# curba ROC
plot_roc_curve(y_test, y_pred, num_classes)

# matricea de confuzie
confusion_matrix_df = plot_confusion_matrix(y_true_classes, y_pred_classes, num_classes=num_classes, class_labels=['0', '1', '2', '3', '4'])

