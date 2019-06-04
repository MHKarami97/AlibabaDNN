import csv
import jdatetime
import calendar
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def GetTestFileData(column):
	result = [] 
	with open('Data/train.csv', 'r') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
		  result.append(row[column])
	del result[0]
	return result

def DateBreaker(input):
	month = []
	day = []
	for i in input:
		datec = jdatetime.datetime.strptime(i,"%Y/%m/%d")
		a=datec.weekday()
		b=datec.strftime("%A")	
		

		date = jdatetime.datetime.strptime(i,"%Y/%m/%d")
		month.append(date.month)
		day.append(date.day)
	#print(month)
	return day,month	

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
	
	DateBreaker(GetTestFileData(1))
	print('finish')