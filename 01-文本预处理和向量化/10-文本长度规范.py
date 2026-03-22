from tensorflow.keras.preprocessing import sequence
x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
             [2, 32, 1, 23, 1]]
# maxlen: 填充或截断序列的最大长度
# post  填充，截断方式后面时，丢弃后面的序列，填充是从后面进行填充
print(sequence.pad_sequences(x_train, maxlen=8, padding='post', truncating='post'))
# pre   填充，截断方式前面时，丢弃前面的序列，填充是从前面进行填充
print(sequence.pad_sequences(x_train, maxlen=8, padding='pre', truncating='pre'))