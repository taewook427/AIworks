import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 기본 경로 설정
# 미니 데이터
base_dir = './chest_xray_mini'
# 본 데이터 
# base_dir = '.../chest_xray/chest_xray'

sub_dirs = ['train', 'test', 'val']
categories = ['PNEUMONIA_VIRUS', 'PNEUMONIA_BACTERIA', 'NORMAL']

# 파일 개수 초기화
file_counts = {
    'train': {'PNEUMONIA_VIRUS': 0, 'PNEUMONIA_BACTERIA': 0, 'NORMAL': 0},
    'test': {'PNEUMONIA_VIRUS': 0, 'PNEUMONIA_BACTERIA': 0, 'NORMAL': 0},
    'val': {'PNEUMONIA_VIRUS': 0, 'PNEUMONIA_BACTERIA': 0, 'NORMAL': 0}
}

# 각 디렉토리와 카테고리의 파일 개수 세기
for sub_dir in sub_dirs:
    for category in categories:
        dir_path = os.path.join(base_dir, sub_dir, category)
        if os.path.exists(dir_path):
            file_counts[sub_dir][category] = len(os.listdir(dir_path))

# 결과 출력
for sub_dir in sub_dirs:
    print(f"{sub_dir.upper()} 데이터셋:")
    for category in categories:
        print(f"  {category}: {file_counts[sub_dir][category]}개")

print("파일 개수 확인이 완료되었습니다.")

# 이미지 경로 설정
img_dir = './chest_xray_mini/train/NORMAL'  # 예시 경로

# 디렉토리에서 이미지 파일 목록 가져오기
img_files = []
for f in os.listdir(img_dir):  # 디렉토리 내의 모든 파일 및 디렉토리 목록 가져오기
    if os.path.isfile(os.path.join(img_dir, f)):  # 파일인지 확인
        img_files.append(f)  # 파일인 경우 리스트에 추가

# 이미지를 5장만 선택
selected_images = img_files[:5]

# 이미지 읽기 및 shape 확인
plt.figure(figsize=(20, 4))  # 전체 그림의 크기 설정 (가로 20인치, 세로 4인치)

for idx, img_file in enumerate(selected_images):  # 선택한 각 이미지 파일에 대해 반복
    img_path = os.path.join(img_dir, img_file)  # 이미지 파일 경로 설정
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 흑백으로 읽어오기
    
    if img is not None:  # 이미지가 제대로 읽혔는지 확인
        # 이미지 shape 출력
        print(f"Image {idx + 1} shape: {img.shape}")

        # 이미지 표시
        plt.subplot(1, 5, idx + 1)  # 1행 5열의 서브플롯 생성
        plt.imshow(img, cmap='gray')  # 이미지를 흑백으로 표시
        plt.title(f"Image {idx + 1}")  # 각 이미지에 제목 붙이기
        plt.axis('off')  # 축을 표시하지 않음
    else:
        print(f"Failed to read image: {img_path}")  # 이미지 읽기 실패 시 경고 메시지 출력

plt.tight_layout()  # 서브플롯 간의 레이아웃을 자동으로 조정
plt.show()  # 모든 이미지를 화면에 표시

# 이미지 크기를 설정합니다. (150 x 150 픽셀로 리사이징)
img_size = 150

# 클래스 라벨을 정의합니다.
labels = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']

# 학습, 검증, 테스트 데이터를 로드하고 전처리하는 함수입니다.
def load_and_preprocess_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)  # 클래스 인덱스를 설정합니다.
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path.replace("\\", "/"), cv2.IMREAD_GRAYSCALE)  # 이미지를 흑백으로 읽어옵니다.
                if img_arr is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # 이미지를 설정된 크기로 리사이즈합니다.
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return data

# 데이터 로드 및 학습, 검증, 테스트 데이터 분리
train_data = load_and_preprocess_data('./chest_xray_mini/train')
val_data = load_and_preprocess_data('./chest_xray_mini/val')
test_data = load_and_preprocess_data('./chest_xray_mini/test')

# 데이터와 라벨을 분리하는 함수입니다.
def split_data_and_labels(data):
    X = []
    y = []
    for feature, label in data:
        X.append(feature)
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = split_data_and_labels(train_data)
X_val, y_val = split_data_and_labels(val_data)
X_test, y_test = split_data_and_labels(test_data)

# 데이터를 0-1 범위로 정규화하는 함수입니다.
def normalize_data(X):
    X_normalized = X / 255.0
    return X_normalized

# 독립변수를 정규화합니다.
X_train_normalized = normalize_data(X_train)
X_val_normalized = normalize_data(X_val)
X_test_normalized = normalize_data(X_test)

# reshape() 의 -1 : 원래의 데이터 수를 유지해주라는 의미로 -1 을 사용합니다.
# reshape() 의 마지막 1 : 흑백 차원을 유지하고 있으므로, 흑백이미지를 의미하는 1차원 을 정의합니다.
# 이미지 데이터를 4차원 텐서로 reshape 합니다. (batch size, height, width, channels)
X_train_normalized = X_train_normalized.reshape(-1, img_size, img_size, 1)
X_val_normalized = X_val_normalized.reshape(-1, img_size, img_size, 1)
X_test_normalized = X_test_normalized.reshape(-1, img_size, img_size, 1)

# 종속변수를 원-핫 인코딩합니다.
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# ImageDataGenerator 객체 생성
# 데이터 증강을 위한 다양한 설정을 지정합니다.
datagen = ImageDataGenerator(
    rotation_range=30,         # 이미지를 최대 30도까지 무작위로 회전시킵니다.
    zoom_range=0.2,            # 이미지를 최대 20%까지 무작위로 확대/축소합니다.
    width_shift_range=0.1,     # 이미지를 좌우로 최대 10%까지 무작위로 이동시킵니다.
    height_shift_range=0.1,    # 이미지를 상하로 최대 10%까지 무작위로 이동시킵니다.
    horizontal_flip=True       # 이미지를 무작위로 좌우 반전시킵니다.
)

# X_train_normalized 데이터에 데이터 증강 적용
# (학습 데이터셋에 지정된 데이터 증강 설정을 적용)
datagen.fit(X_train_normalized)

# 데이터 증강 후의 샘플 이미지 확인
# 데이터 증강이 적용된 이미지를 시각화하여 확인합니다.
for x_batch, y_batch in datagen.flow(X_train_normalized, y_train, batch_size=9):
    # 첫 번째 배치에서 9개의 이미지를 추출하여 시각화합니다.
    for i in range(0, 9):
        # 3x3 그리드의 서브플롯에 이미지를 시각화합니다.
        plt.subplot(330 + 1 + i)
        plt.imshow(x_batch[i].reshape(img_size, img_size), cmap=plt.get_cmap('gray'))
    plt.show()
    break  # 한 배치만 시각화하고 루프를 종료합니다.

# 합성곱 신경망 모델을 정의합니다.
model = Sequential()

# 첫 번째 합성곱 레이어 추가
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))  # 최대 풀링 레이어 추가

# 두 번째 합성곱 레이어 추가
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))  # 필터 개수를 64로 증가
model.add(Dropout(0.1))  # 드롭아웃 레이어 추가 (0.1 비율로 노드 비활성화)
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(MaxPool2D((2, 2), strides=2, padding='same'))  # 최대 풀링 레이어 추가

# 세 번째 합성곱 레이어 추가
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))  # 필터 개수 유지 (64)
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(MaxPool2D((2, 2), strides=2, padding='same'))  # 최대 풀링 레이어 추가

# 네 번째 합성곱 레이어 추가
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))  # 필터 개수를 128로 증가
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가 (0.2 비율로 노드 비활성화)
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(MaxPool2D((2, 2), strides=2, padding='same'))  # 최대 풀링 레이어 추가

# 다섯 번째 합성곱 레이어 추가
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))  # 필터 개수를 256으로 증가
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가 (0.2 비율로 노드 비활성화)
model.add(BatchNormalization())  # 배치 정규화 레이어 추가
model.add(MaxPool2D((2, 2), strides=2, padding='same'))  # 최대 풀링 레이어 추가

# 평탄화 레이어 추가 (2D 데이터를 1D로 변환)
model.add(Flatten())

# 완전 연결 레이어 추가
model.add(Dense(units=128, activation='relu'))  # 은닉층에 128개의 노드 추가
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가 (0.2 비율로 노드 비활성화)

# 출력 레이어 추가 (3개의 클래스로 분류하기 위해 'softmax' 활성화 함수 사용)
model.add(Dense(units=3, activation='softmax'))  # 출력층에 3개의 노드 추가 (각 클래스에 대한 확률 예측)

# 모델 컴파일
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])  # RMSprop 최적화 기법 사용, 손실함수는 categorical_crossentropy, 평가 지표는 정확도

# 학습률 감소 콜백 설정
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)  # 학습률 감소 콜백 설정 (검증 정확도를 모니터링하여 학습률 조정)

model.summary()

# 모델 학습
# datagen.flow()를 통해 데이터 증강이 적용된 배치를 생성하고, 이를 모델에 공급하여 학습합니다.
history = model.fit(
    datagen.flow(X_train_normalized, y_train, batch_size=32),  # 배치 크기 32로 학습 데이터를 증강하여 생성합니다.
    epochs=30,  # 총 30 에포크 동안 학습합니다.
    validation_data=(X_val_normalized, y_val),  # 검증 데이터로 검증을 수행합니다.
    callbacks=[learning_rate_reduction]  # 학습률 감소 콜백을 사용하여 학습률을 조정합니다.
)

# 학습 과정에서 기록된 정확도 및 손실을 시각화하는 함수
def plot_training_history(history):
    # 정확도 시각화
    plt.figure(figsize=(14, 5))
    
    # 학습 정확도
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
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

# 학습 과정에서 기록된 정확도 및 손실 시각화
plot_training_history(history)

model.save('./chest_xray_model.h5')
print("Model saved successfully.")

