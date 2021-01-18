import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow import keras


def save(path, json):
    with open(path, "w") as file:
        file.write(json)


def saveNp(path, arr):
    np.save(path, arr)


def saveTokenizer(path, tokenizer):
    data = tokenizer.to_json()
    save(path, data)


def saveKerasModel(path, model):
    model.save(path)


def restore(path):
    with open(path, mode = 'rt', encoding = 'utf-8') as file:
        return file.read()


def restoreNp(path):
    return np.load(path + '.npy', allow_pickle = True)


def restoreTokenizer(path):
    return tokenizer_from_json(restore(path))


def restoreKerasModel(path):
    keras.models.load_model(path)
