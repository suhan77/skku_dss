import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import *

import KoBertTokenizer as kbt
from utils import save

X_train = save.restoreNp('./result/X_train')
X_test = save.restoreNp('./result/X_test')
tokenizer = save.restoreTokenizer('./result/tokenizer.txt')
print(type(X_train))
print(X_train)
print(X_test)
print(tokenizer)

tokenizer = kbt.KoBertTokenizer.from_pretrained('monologg/kobert')


# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

def convert_data(data_df):
    global tokenizer

    SEQ_LEN = 512  # SEQ_LEN : 버트에 들어갈 인풋의 길이

    tokens, masks, segments, targets = [], [], [], []

    for i in tqdm(range(len(data_df))):
        # token : 문장을 토큰화함
        # token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, pad_to_max_length=True)
        token = tokenizer.tokenize(data_df[DATA_COLUMN][i], max_length = SEQ_LEN, pad_to_max_length = True)
        token = tokenizer.convert_tokens_to_ids(token)

        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros

        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0] * SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)

        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABEL_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets


def convert_data2(data_x, data_y):
    global tokenizer
    global max_len

    SEQ_LEN = max_len  # SEQ_LEN : 버트에 들어갈 인풋의 길이

    tokens, masks, segments, targets = [], [], [], []

    for i in tqdm(range(len(data_x))):
        # token : 문장을 토큰화함
        # token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, pad_to_max_length=True)
        token = data_x[i]

        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일

        num_zeros = (token == 0).sum()  # token.count(0)
        # print(num_zeros, token)
        mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros

        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0] * SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)

        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(np.argmax(data_y[i]))

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets


# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y


# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data2(data_x, data_y):
    data_x, data_y = convert_data2(data_x, data_y)
    return data_x, data_y

path = "./result/model/bert"

# SEQ_LEN = max_len #512
SEQ_LEN = 512
BATCH_SIZE = 20
# 긍부정 문장을 포함하고 있는 칼럼
DATA_COLUMN = "data"
# 긍정인지 부정인지를 (1=긍정,0=부정) 포함하고 있는 칼럼
LABEL_COLUMN = "category"

# train 데이터를 버트 인풋에 맞게 변환
train_x, train_y = load_data(X_train)
# train_x, train_y = load_data2(pad_X_train, del_y_train)
# train_x, train_y = pad_X_train, del_y_train


arr = np.array([0.0, 1.0, 0.0])
print(np.argmax(arr))

arr = np.array([1.0, 0.0, 0.0])
print(np.argmax(arr))

arr = np.array([0.0, 0.0, 1.0])
print(np.argmax(arr))

# TPU 객체 만들기
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


def create_sentiment_bert():
    # 버트 pretrained 모델 로드
    model = TFBertModel.from_pretrained("monologg/kobert", from_pt = True)
    # 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
    token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype = tf.int32, name = 'input_word_ids')
    mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype = tf.int32, name = 'input_masks')
    segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype = tf.int32, name = 'input_segment')
    # 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
    bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
    dnn_units = 256  # 256
    DROPOUT_RATE = 0.2

    bert_outputs = bert_outputs[1]
    # sentiment_first = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
    mid_layer = tf.keras.layers.Dense(dnn_units, activation = 'relu',
                                      kernel_initializer = tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
    mid_layer2 = tf.keras.layers.Dropout(rate = DROPOUT_RATE)(mid_layer)
    sentiment_first = tf.keras.layers.Dense(3, activation = 'softmax',
                                            kernel_initializer = tf.keras.initializers.TruncatedNormal(0.02))(
        mid_layer2)

    sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
    # 옵티마이저는 간단하게 Adam 옵티마이저 활용
    sentiment_model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001),
                            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                            metrics = ['sparse_categorical_accuracy'])
    return sentiment_model


print(train_x)
print(train_y)

num_epochs = 3
batch_size = 20
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# TPU를 활용하기 위해 context로 묶어주기
with strategy.scope():
    sentiment_model = create_sentiment_bert()
    sentiment_model.fit(train_x, train_y, epochs = num_epochs, shuffle = False, batch_size = batch_size)
    sentiment_model.save_weights(os.path.join(path, "sentiment_model.h5"))

def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []

    for i in tqdm(range(len(data_df))):
        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length = SEQ_LEN, pad_to_max_length = True)
        num_zeros = token.count(0)
        mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros
        segment = [0] * SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]


SEQ_LEN = 512
DATA_COLUMN = 'data'


# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x


import shutil

if "bert" not in os.listdir():
    os.makedirs("bert")
else:
    pass


def copytree(src, dst, symlinks = False, ignore = None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


copytree(path, "bert")

test_set = predict_load_data(X_test)

import sys

mod = sys.modules[__name__]
strategy = tf.distribute.experimental.TPUStrategy(resolver)
# TPU를 활용하기 위해 context로 묶어주기
with strategy.scope():
    sentiment_model = create_sentiment_bert()

    sentiment_model.load_weights(os.path.join("bert", "sentiment_model.h5"))
    setattr(mod, 'model', sentiment_model)
    setattr(mod, 'pred_', sentiment_model.predict(test_set, batch_size = 1))

def mean_answer_label(*preds):
    preds_sum = np.zeros(preds[0].shape[0])
    for pred in preds:
        preds_sum += np.argmax(pred, axis = -1)
    return np.round(preds_sum / len(preds), 0).astype(int)

pred_ = np.load(os.path.join(path, 'bert.npz.npy'))

# submission['category'] = mean_answer_label(pred_)
print(pred_)

np.save(os.path.join(path, 'bert.npz'), pred_)


# pred1 = model1.predict(pad_X_test)
#
# pred1
#
# mean_pred = .1 * pred1 + .1 * pred2 + .1 * pred3 + .3 * pred4 + .4 * pred_
#
# submission.category = np.argmax(mean_pred, axis = -1)
#
# submission.to_csv(os.path.join(path, "ens_result.csv"), index = False)

