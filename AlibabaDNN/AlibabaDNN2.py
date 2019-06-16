import jdatetime
import numpy as np
import pandas as pd
from copy import deepcopy
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

witchFile = ''


def ReadFile(status):
    global witchFile

    if status == 1:
        data = pd.read_csv('train.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=10000)

        newData = pd.DataFrame(data)
        newData['newColumn'] = 0
        newData.to_csv('train1.csv')

        witchFile = 'train1'

        return data.groupby(['Log_Date', 'FROM', 'TO']).size()
    else:
        data = pd.read_csv('test.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=10000)

        newData = pd.DataFrame(data)
        newData['newColumn'] = 0
        newData.to_csv('Data/test1.csv')

        witchFile = 'Data/test1.csv'

        return newData, newData.groupby(['Log_Date', 'FROM', 'TO']).size()


def WriteToCSV(csv, row, column, value):
    csv.at[row, column] = value
    csv.to_csv(witchFile + '.csv', index=False)


def preprocess_train(data, label):
    global temp

    train_label_titles = label.index.values

    dic = {}
    for j in range(0, len(label)):
        dic.__setitem__(train_label_titles[j], label[j])

    train_data = deepcopy(data.values)

    i = 0
    train_label = np.array([])
    temp = np.resize(train_data, (10000, 6))
    # temp = np.resize(train_data, (1042617, 8))

    for td in train_data:
        check = td[1:].astype('float32')
        if not np.isnan(check).any():
            label = (td[0], td[2], td[3])
            train_label = np.append(train_label, dic[label])

            date = jdatetime.datetime.strptime(td[0], "%Y/%m/%d")
            d = date.day
            m = date.strftime("%m")
            w = date.weekday()
            s = (int(m) % 12 + 3) // 3

            temp[i][0] = d
            temp[i][1] = int(m)
            temp[i][2] = w
            temp[i][3] = s
            temp[i][4] = td[2]
            temp[i][5] = td[3]

            i += 1
        else:
            temp = np.delete(temp, i, axis=0)

    temp = temp.astype('float32')
    train_label = train_label.astype('float32')

    day = keras.utils.to_categorical(temp[:, 0], 31)
    month = keras.utils.to_categorical(temp[0:, 1], 12)
    weekday = keras.utils.to_categorical(temp[1:, 2], 7)
    season = keras.utils.to_categorical(temp[2:, 3], 4)
    source = keras.utils.to_categorical(temp[3:, 4])
    destination = keras.utils.to_categorical(temp[4:, 5])

    df = pd.concat([pd.DataFrame(data=day), pd.DataFrame(data=month), pd.DataFrame(data=weekday), pd.DataFrame(data=season),
         pd.DataFrame(data=source), pd.DataFrame(data=destination)], axis=1)

    train_label = keras.utils.to_categorical(train_label)

    return df.values, train_label


def preprocess_test(data):
    global temp

    test_data = deepcopy(data.values)

    i = 0
    temp = np.resize(test_data, (10000, 6))
    # temp = np.resize(train_data, (1042617, 8))

    for td in test_data:
        check = td[1:].astype('float32')
        if not np.isnan(check).any():
            date = jdatetime.datetime.strptime(td[0], "%Y/%m/%d")
            d = date.day
            m = date.strftime("%m")
            w = date.weekday()
            s = (int(m) % 12 + 3) // 3

            temp[i][0] = d
            temp[i][1] = int(m)
            temp[i][2] = w
            temp[i][3] = s
            temp[i][4] = td[1]
            temp[i][5] = td[2]

            i += 1
        else:
            temp = np.delete(temp, i, axis=0)

    temp = temp.astype('float32')

    day = keras.utils.to_categorical(temp[:, 0], 31)
    month = keras.utils.to_categorical(temp[0:, 1], 12)
    weekday = keras.utils.to_categorical(temp[1:, 2], 7)
    season = keras.utils.to_categorical(temp[2:, 3], 4)
    source = keras.utils.to_categorical(temp[3:, 4])
    destination = keras.utils.to_categorical(temp[4:, 5])

    df = pd.concat([pd.DataFrame(data=day), pd.DataFrame(data=month), pd.DataFrame(data=weekday), pd.DataFrame(data=season),
         pd.DataFrame(data=source), pd.DataFrame(data=destination)], axis=1)

    return df.values


def create_model():
    model = Sequential()
    model.add(Dense(700, activation='relu', input_shape=(208,)))
    # model.add(Dropout(0.2))
    model.add(Dense(720, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(92, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    model.summary()

    return model


def train_model(model):
    history = model.fit(train[:9000], train_label[:9000],
                        batch_size=100,
                        validation_split=0.2,
                        epochs=20,
                        verbose=2,
                        validation_data=(train[9000:10000], train_label[9000:10000]))

    score = model.evaluate(train[9000:10000], train_label[9000:10000], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def Draw():
    for x in range(5000, 6000):
        test_data = train[x,:].reshape((1, 208))
        predicted_number = model.predict(test_data).argmax()
        label = train_label[x].argmax()
        if (predicted_number == label):
            print('Prediction: %d Label: %d' % (predicted_number, label))
            

if __name__ == '__main__':
    status = 1
    # train_label = ReadFile(status)
    train_data = pd.read_csv('Data/train.csv', sep=',', usecols=[1, 2, 3, 4, 5, 7], nrows=10000)
    train_label = train_data.groupby(['Log_Date', 'FROM', 'TO']).size()
    train, train_label = preprocess_train(train_data, train_label)

    test_data = pd.read_csv('Data/test.csv', sep=',', low_memory=0, usecols=[0, 1, 2], nrows=10000)
    test = preprocess_test(test_data)

    model = create_model()

    train_model(model)

    Draw()

    print('finish')
