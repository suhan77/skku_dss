import datetime as pydatetime

import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# 3. 데이터를 불러옵니다.
train = pd.read_csv('./data/train.csv', encoding = 'utf-8')
test = pd.read_csv('./data/test.csv', encoding = 'utf-8')
sample_submission = pd.read_csv('./data/sample_submission.csv', encoding = 'utf-8')

print(train.shape)
print(test.shape)
print(sample_submission.shape)

# 4. 데이터를 전처리합니다.
train = train.dropna(how = 'any')
train['data'] = train['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test['data'] = test['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '을']

# for test
# train = train.head(5)
# test = test.head(5)
# sample_submission = sample_submission.head(5)

okt = Okt()

# baseline 코드라서 그런지 stopwords 종류가 너무 적고, 아래코드가 너무 비효율적이어서, 특정 품사만 사용하도록 변경이 필요할 것 같다.
X_train = []
for sentence, i in zip(train['data'], tqdm(range(len(train['data'])))):
    temp_X = []
    temp_X = okt.morphs(sentence, stem = True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)

X_test = []
for sentence, i in zip(test['data'], tqdm(range(len(test['data'])))):
    temp_X = []
    temp_X = okt.morphs(sentence, stem = True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)

# 자연어 처리 - https://codetorial.net/tensorflow/natural_language_processing_in_tensorflow_01.html
vocab_size = 30000
tokenizer = Tokenizer(vocab_size, oov_token = "<OOV>")  # 추가 - 토큰화되지 않은 단어 처리하기
# 문자 데이터를 입력받아서 리스트의 형태로 변환
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)
print(X_train)
# 단어들을 시퀀스의 형태로 변환
# 토큰화되어 있지 않은 단어들은 시퀀스에 포함되지 않음
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train)

# 모든 길이를 동일하게 padding을 준다.
max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
print(X_train)

# to binary class matrix
# one-hot encoding
y_train = to_categorical(train['category'])
print(y_train)

# 5. 모델을 생성 및 훈련합니다.
model = Sequential()
model.add(Embedding(vocab_size, 120))
model.add(LSTM(120))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

history = model.fit(X_train, y_train, batch_size = 128, epochs = 15)

# 6. 훈련된 모델로 예측, submission 파일을 생성합니다.
y_pred = model.predict_classes(X_test)
sample_submission['category'] = y_pred

sample_submission.to_csv(f'./result/submission_{pydatetime.datetime.now().timestamp()}.csv',
                         encoding = 'utf-8',
                         index = False)
