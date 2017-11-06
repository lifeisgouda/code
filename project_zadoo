"""
훈련 집합
1 ~ 132		자두	[1,0]
133 ~ 150	잡음	[0,1]

(
133, 134	기타 자두
135 ~ 144	백색소음
145 ~ 150	기타
)


테스트 집합
151 ~ 190	자두	[1,0]
191 ~ 200	기타	[0,1]

"""


import glob
import librosa
import numpy as np
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
from sklearn import linear_model

# 데이터 읽어오기. wav 파일이 여려 개이므로 이를 리스트에 담는다.
y = []
sr = []
for i in range(200):
    file = str(i + 1) + '.wav'

    yt, srt = librosa.load(file)
    y.append(yt)
    sr.append(srt)

# mfcc 변환. 오디오 신호를 mfcc로 바꾼다.
mfcc = []

# n_mfcc를 784가 아닌 25로 한다.
for i in range(200):
    mfcc.append(librosa.feature.mfcc(y=y[i], sr=sr[i], n_mfcc=25))

# shape 확인을 위한 과정
# print("mfcc.shape : ", mfcc.shape) 를 실행하면 AttributeError: 'list' object has no attribute 'shape'
print("mfcc len : ", len(mfcc))
print("mfcc[0].shape : ", mfcc[0].shape)

# 200행의 입력 파일이 존재, 25열의 변수, 각 변수 하나는 31개의 수치로 구성.

# 데이터 처리를 용이하게 하기 위해 평균을 사용한다.
# 31개로 이루어진 부분을 평균을 내야 한다.

mfccMean = []
for i in range(200):
    mfccMean.append(np.mean(mfcc[i], axis = 1))

# mfccMean의 shape을 확인하기.

mfccMeanArray = np.asarray(mfccMean)
print("mfccMeanArray shape : ", mfccMeanArray.shape)
print("mfccMeanArray[0][0] :", mfccMeanArray[0][0])



# train 집합과 test 집합을 만들자.
# 총 200개의 파일이 있다. 150개를 훈련 집합으로, 50개를 테스트 집합으로 정하자.
# 훈련 집합에서 1 ~ 132의 레이블링은 [1,0]이다. 133 ~ 150의 레이블링은 [0,1]이다.
# 테스트 집합에서 151 ~ 190의 레이블링은 [1,0]이다. 191 ~ 200은 [0,1]이다.
# [True, False] 형태로 logits를 구성한다.

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(150):
    x_train.append(mfccMean[i])
    if i < 132:
        y_train.append([1, 0])
    else:
        y_train.append([0, 1])

for i in range(150, 200):
    x_test.append(mfccMean[i])
    if i < 190:
        y_test.append([1, 0])
    else:
        y_test.append([0, 1])



# list를 array로
x_train_array = np.asarray(x_train)
y_train_array = np.asarray(y_train)
x_test_array = np.asarray(x_test)
y_test_array = np.asarray(y_test)


x_train_array = x_train_array.reshape(-1, 5, 5, 1)
x_test_array = x_test_array.reshape(-1, 5, 5, 1)


print("x_train_array : ", x_train_array.shape)
print("y_train_array : ", y_train_array.shape)
print("x_test_array : ", x_test_array.shape)
print("y_test_array : ", y_test_array.shape)




import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 5, 5, 1])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

# W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
# L1 Conv shape=(?, 28, 28, 32)
#    Pool     ->(?, 14, 14, 32)

# L1 Conv shape (?, 5, 5, 32)     Pool (?, 3, 3, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob)


# L2 Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
# W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기

# L2 Conv shape (?, 3, 3, 64)   Pool (?, 2, 2, 64)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob)

# FC 레이어: 입력값 7x7x64 -> 출력값 256
# Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줌.
#    Reshape  ->(?, 256)

# 2 x 2 x 64 = 320 -> 20
W3 = tf.Variable(tf.random_normal([2 * 2 * 64, 20], stddev=0.01))
L3 = tf.reshape(L2, [-1, 2 * 2 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 2개의 출력값을 만듬.

# L3에서 출력 20개를 입력으로 받아서 2개의 출력값을 만듬.

W4 = tf.Variable(tf.random_normal([20, 2], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 10
total_batch = int( len(x_train_array)  / batch_size)

for epoch in range(15):
    total_cost = 0


    for i in range(total_batch):
        batch_xs = []
        batch_ys = []

        for j in range(batch_size):
            batch_xs.append(x_train_array[i*10 + j])
            batch_ys.append(y_train_array[i*10 + j])

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys,
                                              keep_prob: 0.7})

            total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print("model shape: ", model.shape)
print("Y.shape: ", Y.shape)
print('최적화 완료!')


# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: x_test_array, Y: y_test_array, keep_prob: 1}))
