# *_new 데이터 필요

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 형태 변형 (train, val, test)
X0, X1, X2 = np.array( train_new['force'].tolist() ), np.array( val_new['force'].tolist() ), np.array( test_new['force'].tolist() )
y0, y1, y2 = np.array( train_new['updrs'] ), np.array( val_new['updrs'] ), np.array( test_new['updrs'] )
scaler = StandardScaler()
X0 = scaler.fit_transform( X0.reshape( -1, X0.shape[-1] ) ).reshape(X0.shape)
X1 = scaler.fit_transform( X1.reshape( -1, X1.shape[-1] ) ).reshape(X1.shape)
X2 = scaler.fit_transform( X2.reshape( -1, X2.shape[-1] ) ).reshape(X2.shape)

# *_new 데이터 필요

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 형태 변형 (train, val, test)
X0, X1, X2 = np.array( train_new['force'].tolist() ), np.array( val_new['force'].tolist() ), np.array( test_new['force'].tolist() )
y0, y1, y2 = np.array( train_new['updrs'] ), np.array( val_new['updrs'] ), np.array( test_new['updrs'] )
scaler = StandardScaler()
X0 = scaler.fit_transform( X0.reshape( -1, X0.shape[-1] ) ).reshape(X0.shape)
X1 = scaler.fit_transform( X1.reshape( -1, X1.shape[-1] ) ).reshape(X1.shape)
X2 = scaler.fit_transform( X2.reshape( -1, X2.shape[-1] ) ).reshape(X2.shape)

# 압력-중증도 1D CNN 회귀모형
def model_1dcnn(filters, kernel, drop, nodes):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu', input_shape=(X0.shape[1], X0.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drop))

    model.add(Conv1D(filters=filters, kernel_size=kernel-4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drop))

    model.add(Conv1D(filters=filters, kernel_size=kernel-8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Use mean squared error for regression
    history = model.fit(X0, y0, epochs=25, batch_size=32, validation_data=(X1, y1))

    loss, mae = model.evaluate(X2, y2)
    return history, loss, mae

# 하이퍼파라미터 튜닝
temp0 = [ ] # print
temp1 = [ ] # mae
temp2 = [ ] # history
for filters in [32, 64]:
    for kernel in [12, 16]:
            for drop in [0.1, 0.3]:
                for nodes in [50, 100]:
                    history, _, mae = model_1dcnn(filters, kernel, drop, nodes)
                    temp0.append(f"condition : filter={filters}, kernel={kernel}, drop={drop}, nodes={nodes} ==> result : {mae}")
                    temp1.append(mae)
                    temp2.append(history)

# 학습 과정에서 기록된 정확도 및 손실을 시각화하는 함수
def plot_training_history(history):
    # 정확도 시각화
    plt.figure(figsize=(14, 5))

    # 학습 정확도 (오차)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training Error Mean')
    plt.plot(history.history['val_mae'], label='Validation Error Mean')
    plt.title('Training and Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()

    # 손실 시각화
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

print( "\n".join(temp0) )
minpos = np.argmin(temp1)
print( "minimal error at : ", temp0[minpos] )
plot_training_history( temp2[minpos] )

# 학습 과정에서 기록된 정확도 및 손실을 시각화하는 함수
def plot_training_history(history):
    # 정확도 시각화
    plt.figure(figsize=(14, 5))

    # 학습 정확도 (오차)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training Error Mean')
    plt.plot(history.history['val_mae'], label='Validation Error Mean')
    plt.title('Training and Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()

    # 손실 시각화
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

print( "\n".join(temp0) )
minpos = np.argmin(temp1)
print( "minimal error at : ", temp0[minpos] )
plot_training_history( temp2[minpos] )
