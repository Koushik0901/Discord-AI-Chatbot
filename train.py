import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Dropout,
    LayerNormalization,
    Bidirectional,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

try:
    model = keras.models.load_model("./chat_model.h5")
    print("Already Trained!")
except:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            training_sentences.append(pattern)
            training_labels.append(intent["tag"])
        responses.append(intent["responses"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    num_classes = len(labels)
    training_labels = np.array(training_labels).reshape(-1, 1)

    lbl_encoder = OneHotEncoder(sparse_output=False)
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 128
    max_len = 25
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(
        sequences, truncating="post", maxlen=max_len, padding="post"
    )

    # creating a bidirectional lstm model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(32, kernel_regularizer="l2")))
    model.add(LayerNormalization())
    model.add(Dense(64, activation="gelu", kernel_regularizer="l2"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            name="categorical_crossentropy",
        ),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    print(model.summary())

    epochs = 50
    history = model.fit(
        padded_sequences,
        training_labels,
        epochs=epochs,
    )

    os.makedirs("checkpoints", exist_ok=True)
    # to save the trained model
    model.save("checkpoints/chat_model.h5")

    import pickle

    # to save the fitted tokenizer
    with open("checkpoints/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open("checkpoints/label_encoder.pickle", "wb") as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
