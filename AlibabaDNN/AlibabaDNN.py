import csv
import jdatetime
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Dense, Dropout

witchFile = ''

def ReadFile(status):
	global witchFile

	if status == 1:
		data = pd.read_csv('Data/train.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=10000)

		newData = pd.DataFrame(data)
		newData['newColumn'] = 0
		newData.to_csv('Data/train1.csv')
						
		witchFile = 'Data/train1'

		return newData,newData.groupby(['Log_Date', 'FROM', 'TO']).size()
	else:
		data = pd.read_csv('Data/test.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=10000)

		newData = pd.DataFrame(data)
		newData['newColumn'] = 0
		newData.to_csv('Data/test1.csv')

		witchFile = 'Data/test1.csv'

		return newData,newData.groupby(['Log_Date', 'FROM', 'TO']).size()

def WriteToCSV(csv,row,column,value):
	csv.at[row, column] = value
	csv.to_csv(witchFile + '.csv', index=False)

def preprocess(train_data,train_label_values):
    global temp
    
    train_label_titles = train_label_values.index.values
    
    dic = {}
    for j in range(0, len(train_label_values)):
        dic.__setitem__(train_label_titles[j], train_label_values[j])    

    train_data = deepcopy(train_data.values)
    
    i = 0
    train_label = np.array([])
    temp = np.resize(train_data, (10000, 9))

    for td in train_data:
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
        temp[i][4] = td[1]
        temp[i][5] = td[2]
        temp[i][6] = td[3]
        temp[i][7] = td[4]
        temp[i][8] = td[5]

        i += 1

    # train_label = keras.utils.to_categorical(train_label_values, 1)
    # test_label = keras.utils.to_categorical(test_label, 10)
    #train_label = train_label.astype('float32')  
    #temp = temp.astype('float32')

    return temp, train_label

def create_model():
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(9,)))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    model.summary()

    return model

def train_model(model):
    #print(train_data[:1000])
    #model.cast(train_label, tf.float32)
    #aa = tf.cast(train_data[:1000], tf.int32)
    #bb = tf.cast(train_label[:1000], tf.int32)
    history = model.fit(train_data[:9000], train_label[:9000],
                        batch_size=100,
                        epochs=10,
                        verbose=2,
                        validation_data=(train_data[9000:10000], train_label[9000:10000]))

    score = model.evaluate(train_data[5000:6000], train_label[5000:6000], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def OneHotEncoding(input):
	values = array(input)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	print(integer_encoded)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	print(onehot_encoded)
	# invert first example
	inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
	print(inverted)

def Draw(train_data,model):
	for x in range(5000, 6000):
		test_data = train_data[x,:].reshape((9,))
		predicted_number = model.predict(test_data).argmax()
		label = train_label[x].argmax()
        # if (predicted_number != label):
            # plt.title('Prediction: %d Label: %d' % (predicted_number, label))
            # plt.imshow(test_data, cmap=plt.get_cmap('gray_r'))
            # plt.show()

if __name__ == '__main__':	
	status = 1
	data,train_label = ReadFile(status)

	train_data, train_label = preprocess(data,train_label)

	model = create_model()

	train_model(model)

	Draw(train_data,model)

	#test
	#WriteToCSV(data,1,'newColumn',2)
	#test

	#day, month, dayOfWeekNum, monthOfYearNum = DateBreaker(data)

	print('finish')