import csv
import jdatetime
import calendar
import numpy as np
import pandas as pd
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

witchFile = ''

def ReadFile(status):
	global witchFile

	if status == 1:
		data = pd.read_csv('Data/train.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=100)

		newData = pd.DataFrame(data)		
		newData['newColumn'] = 0		
		newData.to_csv('Data/train1.csv')
						
		witchFile = 'Data/train1'

		return newData,newData.groupby(['Log_Date', 'FROM', 'TO']).size()
	else:
		data = pd.read_csv('Data/test.csv', sep=',', low_memory=0, usecols=[1, 2, 3, 4, 5, 7], nrows=100)

		newData = pd.DataFrame(data)		
		newData['newColumn'] = 0		
		newData.to_csv('Data/test1.csv')

		witchFile = 'Data/test1.csv'

		return newData,newData.groupby(['Log_Date', 'FROM', 'TO']).size()

def WriteToCSV(csv,row,column,value):
	csv.at[row, column] = value
	csv.to_csv(witchFile + '.csv', index=False)

def DateBreaker(train_data):
	day = []
	dayOfWeekNum = []
	monthOfYearNum = []
	seasonNum = []
	train = train_data.values
	for td in train_data['Log_Date']:
		date = jdatetime.datetime.strptime(td,"%Y/%m/%d")

		d = date.day
		m = date.strftime("%m")
		w = date.weekday()

		s = (int(m) % 12 + 3) // 3
		#train_data['Log_Date'][0] = [d, m, w, s]
		train[0][0] = d
		train[0] = np.append(train[0], m)
		train[0] = np.append(train[0], w)
		train[0] = np.append(train[0], s)

		day.append(d)
		dayOfWeekNum.append(date.weekday(w))
		monthOfYearNum.append(m)
		seasonNum.append(s)
	
	return day, dayOfWeekNum, monthOfYearNum	

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

if __name__ == '__main__':	
	status = 1
	data,train_label = ReadFile(status)

	#test
	WriteToCSV(data,1,'newColumn',2)
	#test

	day, month, dayOfWeekNum, monthOfYearNum = DateBreaker(data)

	model = Sequential()
	model.add(Dense(8, activation = 'relu', input_shape = (10,)))

	print('finish')