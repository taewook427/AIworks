# https://www.kaggle.com/datasets/tovarischsukhov/southparklines
# https://github.com/ukairia777/tensorflow-nlp-tutorial/tree/main/08.%20RNN

"""
with open("All-seasons.csv", "r", encoding="utf-8") as f:
    t = f.read().split("\n")
a = ""
for i in t:
    if '"' in i and len(i) > 2:
        a = a + i[i.index('"')+1:] + " "
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(a)
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed

# 텍스트 파일 로드
def load_text():
    with open('data.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.lower().replace("\t"," ")
    return "".join([x for x in text if ord(x) < 128])

data = load_text()[:4000001] # 텍스트 문자열 로드
char = sorted(list(set(data))) # 글자 1개
idx_tkn, rev_tkn = dict(), dict()
for i in range(0, len(char)):
    idx_tkn[char[i]] = i
    rev_tkn[i] = char[i]
print(idx_tkn, rev_tkn)

# 단어 인코딩 저장
with open('idx_tkn.pkl', 'wb') as file:
    pickle.dump(idx_tkn, file)
with open('rev_tkn.pkl', 'wb') as file:
    pickle.dump(rev_tkn, file)

# 학습 세트 생성 (문장길이 60)
X, y = [ ], [ ]
for i in range(0, len(data)//60):
    i = i * 60
    X.append( np.array( [idx_tkn[x] for x in data[i:i+60]] ) )
    y.append( np.array( [idx_tkn[x] for x in data[i+1:i+61]] ) )

X = to_categorical(X)
y = to_categorical(y)
print(X.shape, y.shape, len(X), len(y))

# Sequential 모델 생성
model = Sequential()
model.add(Input(shape=(None, X.shape[2]))) # 희소 벡터 입력
model.add(LSTM(512, return_sequences=True)) # 512 node LSTM
model.add(Dropout(0.2)) # 드롭아웃 20%
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributed(Dense(len(idx_tkn), activation='softmax')))

# 모델 컴파일, 요약
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 모델 학습
model.fit(X, y, epochs=80, batch_size=128, validation_split=0.2)
model.save_weights('model.weights.h5')

# ------------------------------ 모델 사용 ------------------------------

# 단어 인코딩 정보 불러오기
with open('idx_tkn.pkl', 'rb') as file:
    idx_tkn = pickle.load(file)
with open('rev_tkn.pkl', 'rb') as file:
    rev_tkn = pickle.load(file)

# 동일 구조 모델 생성
model = Sequential()
model.add(Input(shape=(None, len(idx_tkn))))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributed(Dense(len(idx_tkn), activation='softmax')))

# 가중치 로드, 컴파일
model.load_weights('model.weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 문장 생성
def generate_text(seed, n):
    sign = [".", ",", "!", "?"]
    result, dec = [x for x in seed], np.zeros((1, n+len(seed), len(idx_tkn)))
    for i in range(0, len(seed)):
        dec[0][i][idx_tkn[seed[i]]] = 1
    print(seed, end="")
    for i in range(len(seed), n+len(seed)):
        c = np.argmax( model.predict(dec[:, :i, :], verbose=0)[0][-1] )
        dec[0][i][c] = 1
        c = rev_tkn[c]
        result.append(c)
        print(c, end="")
        if c in sign and result[-2] not in sign:
            print("")
    print("")
    return "".join(result)

# 생성 예시
_ = generate_text("bad government", 500)
