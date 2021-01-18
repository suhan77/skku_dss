import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop

from utils import save

np.random.seed(456)

train_data = pd.read_csv('./data/train.csv', encoding = 'utf-8')
test_data = pd.read_csv('./data/test.csv', encoding = 'utf-8')
submission = pd.read_csv('./data/sample_submission.csv', encoding = 'utf-8')

def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',
                        str(texts[i]))  # remove punctuation
        review = re.sub(r'\d+', '', str(texts[i]))  # remove number
        review = review.lower()  # lower case
        review = re.sub(r'\s+', ' ', review)  # remove extra space
        review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
        review = re.sub(r'\s+', ' ', review)  # remove spaces
        review = re.sub(r"^\s+", '', review)  # remove space from start
        review = re.sub(r'\s+$', '', review)  # remove space from the end
        corpus.append(review)
    return corpus


train_data.data = clean_text(train_data.data)
test_data.data = clean_text(test_data.data)

train_data_text = list(train_data['data'])

train_clear_text = []

for i in tqdm(range(len(train_data_text))):
    train_clear_text.append(str(train_data_text[i]).replace('\\n', ''))
train_data['clear_text'] = train_clear_text
train_data.head()

train_clear_text = list(train_data['clear_text'])

train_clear_text2 = []

for text in train_clear_text:
    temp = re.sub('[-=+,#:;//●<>▲\?:^$.☆!★()Ⅰ@*\"※~>`\'…》]', ' ', text)
    train_clear_text2.append(temp)
train_data['clear_text'] = train_clear_text2
train_data.head()

test_data_text = list(test_data['data'])

test_clear_text = []

for i in tqdm(range(len(test_data_text))):
    test_clear_text.append(test_data_text[i].replace('\\n', ' '))
test_data['clear_text'] = test_clear_text
test_data.head()

test_clear_text = list(test_data['clear_text'])

test_clear_text2 = []

for text in test_clear_text:
    temp = re.sub('[-=+,#:;//●<>▲\?:^$.☆!★()Ⅰ@*\"※~>`\'…》]', ' ', text)
    test_clear_text2.append(temp)
test_data['clear_text'] = test_clear_text2
test_data.head()

# Using Mecab for tokenizing
from konlpy.tag import Mecab

# 설치가 복잡해서 mecab말고 다른 라이브러리 사용한다.
# https://joyae.github.io/2020-10-02-Mecab/
# mecab = Mecab()

from konlpy.tag import Okt
okt = Okt()

stop_df = pd.read_csv('./data/한국어불용어100.txt', sep = '\t', header = None, names = ['형태', '품사', '비율'])
stop_df.tail()

stop_df.loc[100] = '가'
stop_df.loc[101] = '합니다'

stop_words = list(stop_df.형태)

# delete outlier data
ind_list = [24885, 14916, 14605, 6641, 17406, 26957, 2175, 6885, 8947, 14966, 8198, 25955, 39167, 21707, 12678,
            3023, 31971, 3730, 37153, 33481, 33369, 12927, 30773, 36431, 12373, 37525, 27530, 8958, 16884, 18072,
            4478, 7940, 16400, 16656]
train_data = train_data.query('index not in @ind_list')
train_data.index = range(0, len(train_data))

X_train = []

train = list(train_data['clear_text'])

for i in tqdm(range(len(train))):
    temp_X = []
    temp_X = okt.nouns(train[i])  # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words]  # 불용어 제거
    temp_X = [word for word in temp_X if len(word) > 1]
    X_train.append(temp_X)

X_test = []

test = list(test_data['clear_text'])

for i in tqdm(range(len(test))):
    temp_X = []
    temp_X = okt.nouns(test[i])  # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words]  # 불용어 제거
    temp_X = [word for word in temp_X if len(word) > 1]
    X_test.append(temp_X)


# modeling
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 축소해서 구하려는 경우 개수를 낮춰야한다.
# threshold = 11
threshold = 1
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

vocab_size = total_cnt - rare_cnt + 1  # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :', vocab_size)

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)

token_X_train = tokenizer.texts_to_sequences(X_train)
token_X_test = tokenizer.texts_to_sequences(X_test)

print(X_train)
print(X_test)

print(token_X_train)
print(token_X_test)

# 시간 절약과 같은 결과를 위해 미리 파일로 저장한다.
# save.saveNp("./result/X_train", X_train)
# save.saveNp("./result/X_test", X_train)
# save.saveTokenizer("./result/tokenizer.txt", tokenizer)

y_train = to_categorical(np.array(train_data['category']))

drop_train = [index for index, sentence in enumerate(token_X_train) if len(sentence) < 1]

# 빈 샘플들을 제거
del_X_train = np.delete(token_X_train, drop_train, axis = 0)
del_y_train = np.delete(y_train, drop_train, axis = 0)
print(len(del_X_train))
print(len(del_y_train))

print('train data의 최대 길이 :', max(len(l) for l in del_X_train))
print('train data의 평균 길이 :', sum(map(len, del_X_train)) / len(del_X_train))
plt.hist([len(s) for s in X_train], bins = 50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

print("test data의 최대 길이 : ", max(len(l) for l in token_X_test))
print("test data의 평균 길이 : ", sum(map(len, token_X_test)) / len(token_X_test))
plt.hist([len(s) for s in token_X_test], bins = 50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (cnt / len(nested_list)) * 100))


max_len = 300
below_threshold_len(max_len, del_X_train)

pad_X_train = pad_sequences(del_X_train, maxlen = max_len)
pad_X_test = pad_sequences(token_X_test, maxlen = max_len)

model1 = Sequential()
model1.add(Embedding(vocab_size, 64, input_length = max_len))
model1.add(Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_regularizer = l2(0.01)))
model1.add(GlobalMaxPooling1D())
model1.add(Dense(3, activation = 'softmax'))
model1.summary()

model1.compile(optimizer = RMSprop(lr = .0005), loss = 'categorical_crossentropy', metrics = ['acc'])
reLR = ReduceLROnPlateau(patience = 5, verbose = 1, factor = .2)
es = EarlyStopping(monitor = 'val_acc', mode = 'max', verbose = 1, patience = 2)
mc = ModelCheckpoint(filepath = './result/model/1028_1.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True,
                     restore_best_weights = True)

history1 = model1.fit(pad_X_train, del_y_train, epochs = 30, batch_size = 64, shuffle = True, validation_split = 0.2,
                      verbose = 1, callbacks = [es, mc, reLR])

model1 = load_model('./result/model/1028_1.h5')  # val_acc = 0.87893

pred1 = model1.predict(pad_X_test)

model2 = Sequential()
model2.add(Embedding(vocab_size, 64, input_length = max_len))
model2.add(Conv1D(64, 5, activation = 'relu', kernel_regularizer = l2(0.001)))
model2.add(MaxPooling1D(5))
model2.add(Dropout(.5))
model2.add(Conv1D(64, 5, activation = 'relu', kernel_regularizer = l2(.001)))
model2.add(GlobalMaxPooling1D())
model2.add(BatchNormalization())
model2.add(Dense(3, activation = 'softmax', kernel_regularizer = l2(0.001)))
model2.summary()

model2.compile(optimizer = RMSprop(lr = .0005), loss = 'categorical_crossentropy', metrics = ['acc'])

callback_ear = [EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min', verbose = 1),
                ModelCheckpoint(filepath = './result/model/1028_2.h5', monitor = 'val_acc', save_best_only = True, mode = 'max',
                                verbose = 1)]
history2 = model2.fit(pad_X_train, del_y_train, epochs = 15, batch_size = 64, verbose = 1, validation_split = .2,
                      callbacks = callback_ear)

model2 = load_model('./result/model/1028_2.h5')  # val_acc = 0.8841

pred2 = model2.predict(pad_X_test)

model3 = Sequential()
model3.add(Embedding(vocab_size, 64, input_length = max_len))
model3.add(Conv1D(32, 5, activation = 'relu'))
model3.add(Conv1D(32, 5, activation = 'relu'))
model3.add(Conv1D(32, 5, activation = 'relu'))
model3.add(MaxPooling1D(pool_size = 4))
model3.add(LSTM(16))
model3.add(Dropout(0.4))
model3.add(Dense(3, activation = 'softmax'))
model3.summary()

model3.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = .0005), metrics = ['acc'])

callback_ear = [EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 2, verbose = 1),
                ModelCheckpoint(filepath = './result/model/1028_3.h5', monitor = 'val_acc', save_best_only = True, mode = 'max',
                                verbose = 1)]

history3 = model3.fit(pad_X_train, del_y_train, epochs = 30, batch_size = 32, validation_split = 0.2, verbose = 1,
                      shuffle = True, callbacks = callback_ear)

model3 = load_model('./result/model/1028_3.h5')

pred3 = model3.predict(pad_X_test)

model4 = Sequential()
model4.add(Embedding(vocab_size, 32, input_length = max_len))
model4.add(Dropout(0.3))
model4.add(Conv1D(32, 5, activation = 'relu'))
model4.add(MaxPooling1D(pool_size = 4))
model4.add(LSTM(32))
model4.add(Dense(3, activation = 'softmax'))
model4.summary()

model4.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = .0005), metrics = ['acc'])

callback_ear = [EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 2, verbose = 1),
                ModelCheckpoint(filepath = './result/model/1028_4.h5', monitor = 'val_acc', save_best_only = True, mode = 'max',
                                verbose = 1)]

history4 = model4.fit(pad_X_train, del_y_train, epochs = 30, batch_size = 64, validation_split = 0.2, verbose = 1,
                      shuffle = True, callbacks = callback_ear)

model4 = load_model('./result/model/1028_4.h5')

pred4 = model4.predict(pad_X_test)

mean_pred = .1 * pred1 + .3 * pred2 + .2 * pred3 + .4 * pred4

submission.category = np.argmax(mean_pred, axis = -1)

submission.to_csv("./result/ens1028.csv", index = False)
