import tensorflow
from tensorflow import keras

print("Keras Version : {}".format(keras.__version__))

import shap

print("SHAP Version : {}".format(shap.__version__))

from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
batchsize = 1000
(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

X_train = X_train[:1000,:,:]
X_test = X_test[:500,:,:]
Y_train = Y_train[:1000]
Y_test  = Y_test[:500]

X_train, X_test = X_train.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)

X_train, X_test = X_train/255.0, X_test/255.0

classes =  np.unique(Y_train)
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
mapping = dict(zip(classes, class_labels))


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"),

    layers.Flatten(),
    layers.Dense(len(classes), activation="softmax")
])

model.summary()

model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
model.fit(X_train, Y_train, batch_size=256, epochs=5, validation_data=(X_test, Y_test))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Y_test_preds = model.predict(X_test)
Y_test_preds = np.argmax(Y_test_preds, axis=1)

print("Test Accuracy : {}".format(accuracy_score(Y_test, Y_test_preds)))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_test, Y_test_preds))
print("\nClassification Report :")
print(classification_report(Y_test, Y_test_preds, target_names=class_labels))

sizeofval = X_train[0].shape
masker = shap.maskers.Image("inpaint_telea", X_train[0].shape)

explainer = shap.Explainer(model, masker, output_names=class_labels)
inputval = X_test[:4]
flipped_value = shap.Explanation.argsort.flip[:5]
shap_values = explainer(X_test[:4], outputs=flipped_value)

print(shap_values.shape)

print("Actual Labels    : {}".format([mapping[i] for i in Y_test[:4]]))
probs = model.predict(X_test[:4])
print("Predicted Labels : {}".format([mapping[i] for i in np.argmax(probs, axis=1)]))
print("Probabilities : {}".format(np.max(probs, axis=1)))

shap.image_plot(shap_values)