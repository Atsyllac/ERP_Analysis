import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras.layers import Add
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


npz = np.load("C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\thealldata.npz") #using absolute path

eeg_test = npz["eeg_test"]
eeg_train = npz["eeg_train"]
sentence_train = npz["sentence_train"]
sentence_test = npz["sentence_test"]
labels_train = npz["labels_train"]
labels_test = npz["labels_test"]

# print(eeg_train.shape, sentence_train.shape, labels_train.shape) #(8929, 63, 1201) (8929, 768, 10) (8929,)
# print(eeg_test.shape, sentence_test.shape, labels_test.shape) #(2233, 63, 1201) (2233, 768, 10) (2233,)

# eeg_train_transposed = np.transpose(eeg_train, (0, 2, 1))
sentence_train_transposed = np.transpose(sentence_train, (0, 2, 1))
sentence_test_transposed = np.transpose(sentence_test, (0, 2, 1))

#normalizing eeg data, preventing 0s
eeg_train_norm = (eeg_train - np.mean(eeg_train, axis=2, keepdims=True)) / (np.std(eeg_train, axis=2, keepdims=True) + 1e-10)
eeg_test_norm  = (eeg_test  - np.mean(eeg_test,  axis=2, keepdims=True)) / (np.std(eeg_test,  axis=2, keepdims=True) + 1e-10)


#-------------CNNs-------------#
eeg_CNN = layer.Input(shape=(63, 1201, 1))
x = layer.Conv2D(filters=32, kernel_size=(5, 15), activation='relu', padding='same')(eeg_CNN)
x = layer.BatchNormalization()(x)
x = layer.AveragePooling2D(pool_size=(3, 1))(x)
x = layer.Conv2D(filters=64, kernel_size=(5, 15), activation='relu', padding='same')(x)
x = layer.BatchNormalization()(x)
x = layer.GlobalAveragePooling2D()(x)
eeg_cnn_output = layer.Dense(256, activation='relu')(x)


#this CNN must use Conv1D or things will be bad (Conv2D scrambles embeddings)
embedding_CNN = layer.Input(shape=(10, 768))
z = layer.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(embedding_CNN)
z = layer.BatchNormalization()(z)
z = layer.Conv1D(128, kernel_size=3, padding="same", activation="relu")(z)
z = layer.GlobalAveragePooling1D()(z)
embedding_cnn_output = layer.Dense(256, activation='relu')(z)



#-------------Gates & Fusion-------------#
# eeg_gate = layer.Dense(256, activation='sigmoid')(eeg_cnn_output)
# gated_eeg = layer.Multiply()([eeg_cnn_output, eeg_gate])
#
# embedding_gate = layer.Dense(256, activation='sigmoid')(embedding_cnn_output)
# gated_embeddings = layer.Multiply()([embedding_cnn_output, embedding_gate])
#
# # fused_data = Add()([gated_eeg, gated_embeddings])
# concat_data = layer.Concatenate()([gated_eeg, gated_embeddings])

concat_data = layer.Concatenate()([eeg_cnn_output, embedding_cnn_output])


#-------------ANN Classifier-------------#
ANN = layer.Dense(256, activation='relu')(concat_data)
w = layer.Dense(256, activation='relu')(ANN)
w = layer.Dropout(0.3)(w)
w = layer.Dense(128, activation='relu')(w)
w = layer.Dropout(0.2)(w)
ANN_output = layer.Dense(4, activation='softmax')(w)


model = tf.keras.Model(inputs=[eeg_CNN, embedding_CNN], outputs=ANN_output)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()


history = model.fit(
    [eeg_train_norm, sentence_train_transposed],
    labels_train,
    batch_size=64,
    epochs=10,
    verbose=2
)

predictions = model.predict([eeg_test_norm, sentence_test_transposed])




#-------------VISUALIZATION-------------#
plt.title("Accuracy over Epochs")
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
#
plt.title("Loss Reduction Over Epochs")
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
#
classes = ["grammatical error", "normal", "semantic error", "syntactic grammatical error"]
max_values = np.squeeze(np.array(predictions.argmax(axis=1)))
cm = confusion_matrix(labels_test, max_values, labels=[0,1,2,3])
fig, ax = plt.subplots(figsize=(13,13))
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(classes,rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(classes,rotation=0, fontsize = 18)

plt.show()
