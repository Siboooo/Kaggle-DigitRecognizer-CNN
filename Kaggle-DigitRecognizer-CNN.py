import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy

x = tf.placeholder(tf.float32, [None, 28*28])
xImg = tf.reshape(x, [-1, 28, 28, 1])

#Layer 1
weightConv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
biasConv1 = tf.Variable(tf.constant(0.1, shape=[32]))
#Convolution [28, 28, 1] --> [28, 28, 32]
activedConv1 = tf.nn.relu(tf.nn.conv2d(xImg, weightConv1, strides=[1, 1, 1, 1], padding="SAME") + biasConv1)
#Pooling [28, 28, 32] --> [14, 14, 32]
maxPool1 = tf.nn.max_pool(activedConv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#Layer 2
weightConv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
biasConv2 = tf.Variable(tf.constant(0.1, shape=[64]))
#Convolution [14, 14, 32] --> [14, 14, 64]
activedConv2 = tf.nn.relu(tf.nn.conv2d(maxPool1, weightConv2, strides=[1, 1, 1, 1], padding="SAME") + biasConv2)
#Pooling [14, 14, 64] --> [7, 7, 64]
maxPool2 = tf.nn.max_pool(activedConv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#Layer 3
weightFullCon1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
biasFullCon1 = tf.Variable(tf.constant(0.1, shape=[1024]))
maxPool2New = tf.reshape(maxPool2, [-1, 7*7*64])
#Full Connection
activedFullCon1 = tf.nn.relu(tf.matmul(maxPool2New, weightFullCon1) + biasFullCon1)

dropP = tf.placeholder(tf.float32)
#Dropout
droppedAFC1 = tf.nn.dropout(activedFullCon1, 1-dropP)

#Layer 4
weightFullCon2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
biasFullCon2 = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.nn.softmax(tf.matmul(droppedAFC1, weightFullCon2) + biasFullCon2)

yOneHot = tf.placeholder(tf.float32, [None, 10])
crossEntropy = -tf.reduce_sum(yOneHot * tf.log(y))
training = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
predict = tf.argmax(y, 1)

trainData = pd.DataFrame(pd.read_csv("train.csv"))
yTrain = trainData["label"]
xTrain = trainData.drop(labels=["label"], axis=1)
xTrain = xTrain/255.0
del trainData

session = tf.Session()
session.run(tf.global_variables_initializer())

batch = 50
batchNum = int(numpy.ceil(xTrain.index.size//batch))
#epoch
for epoch in range(10):
    for i in range(batchNum):
        if i%100 == 0:
            print("Traing --> ", i, "/", batchNum, " in epoch ", epoch)
        yOH = session.run(tf.one_hot(yTrain[i*batch:i*batch+batch], 10))
        session.run(training, feed_dict={x: xTrain[i*batch:(i*batch)+batch], yOneHot: yOH, dropP: 0.5})

print("Loading test dataset.")

testData = pd.read_csv("test.csv")
testData = testData/255.0

total = testData.index.size
result = pd.DataFrame(columns=["Imageid", "Label"])
for i in range(total):
    prediction = predict.eval(session=session, feed_dict={x: testData[i:i+1], dropP: 0})
    temp = pd.Series({"Imageid": i+1, "Label": prediction[0]})
    result = result.append(temp, ignore_index=True)
result.to_csv("DR-CNN-Cus-sub.csv", index=False)


testDataN = pd.read_csv("mnist_test.csv")
yTest = testDataN["7"]
xTest = testDataN.drop(labels=["7"], axis=1)
xTest = xTest/255.0
del testDataN

total = yTest.index.size
correct = 0
for i in range(total):
    prediction = predict.eval(session=session, feed_dict={x: xTest[i:i+1], dropP: 0})
    if prediction[0] - int(yTest[i:i+1]) == 0:
        correct += 1
print("Accuracy: ", correct/total)
