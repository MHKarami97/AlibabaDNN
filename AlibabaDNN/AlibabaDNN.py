import csv
import jdatetime
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from tensorflow import keras
from sklearn import preprocessing
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
		data = pd.read_csv('Data/test.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4], nrows=10000)

		newData = pd.DataFrame(data)
		newData['newColumn'] = 0
		newData.to_csv('Data/test1.csv')

		witchFile = 'Data/test1.csv'

		return newData,newData

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
    history = model.fit(train_data[:9000], train_label[:9000],
                        batch_size=100,
                        epochs=10,
                        verbose=2,
                        validation_data=(train_data[9000:10000], train_label[9000:10000]))

    score = model.evaluate(train_data[5000:6000], train_label[5000:6000], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def OneHotEncoding(data,dataTest):
    # Discrete column to be one-hot-encoded
    #col = 'FROM'
    #train = pd.DataFrame(data)
    #test = pd.DataFrame(dataTest)
    ## Create dummy variables for each level of `col`
    #train_animal_dummies = pd.get_dummies(data[col], prefix=col)
    #train = train.join(train_animal_dummies)

    #test_animal_dummies = pd.get_dummies(dataTest[col], prefix=col)
    #test = test.join(test_animal_dummies)

    ## Find the difference in columns between the two datasets
    ## This will work in trivial case, but if you want to limit to just one
    ## feature
    ## use this: f = lambda c: col in c; feature_difference = set(filter(f,
    ## train)) - set(filter(f, test))
    #feature_difference = set(train) - set(test)

    ## create zero-filled matrix where the rows are equal to the number
    ## of row in `test` and columns equal the number of categories missing
    ## (i.e.  set difference
    ## between relevant `train` and `test` columns
    #feature_difference_df = pd.DataFrame(data=np.zeros((test.shape[0],
    #len(feature_difference))),
    #                                     columns=list(feature_difference))

    ## add "missing" features back to `test
    #test = test.join(feature_difference_df)

    df = pd.DataFrame(data)

    one_hot_from = pd.get_dummies(df['FROM'])
    one_hot_to = pd.get_dummies(df['TO'])

    te = []
    te1 = []
    te2 = []
    te3 = []
    te4 = []
    i = 0
    for tds in data.values:
        date = jdatetime.datetime.strptime(tds[0], "%Y/%m/%d")
        
        d = date.day
        m = date.strftime("%m")
        w = date.weekday()
        s = (int(m) % 12 + 3) // 3

        mytemp = [d,int(m),w,s]
        te.append(mytemp)

        te1.append(d)
        te2.append(int(m))
        te3.append(w)
        te4.append(s)

        i += 1

    one_hot_d = pd.get_dummies(te1)
    one_hot_m = pd.get_dummies(te2)
    one_hot_w = pd.get_dummies(te3)
    one_hot_s = pd.get_dummies(te4)

    arrz = [one_hot_d ,one_hot_m , one_hot_w , one_hot_s]
    one_hot_date = np.array(arrz)

    print(one_hot_date)
    print('-------------------')
    
def Draw(train_data,model):
	for x in range(5000, 6000):
		test_data = train_data[x,:].reshape((9,))
		predicted_number = model.predict(test_data).argmax()
		label = train_label[x].argmax()
        
if __name__ == '__main__':	
    status = 1
    data,train_label = ReadFile(status)
    dataTest,train_labelTest = ReadFile(0)

    OneHotEncoding(data,dataTest)

    train_data, train_label = preprocess(data,train_label)

    model = create_model()

    train_model(model)

    Draw(train_data,model)

    #test
    #WriteToCSV(data,1,'newColumn',2)
    #test

    #day, month, dayOfWeekNum, monthOfYearNum = DateBreaker(data)

    print('finish')