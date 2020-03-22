#script only to get the data for CoronaVirus Cases 
#..All over the world.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import pandas as pd
import numpy as np
import OpenBlender 
import matplotlib
import wordcloud
import json
import os

def FilterDateUntilYear(train_X_copy, val_X_copy):
	train_X_list = []
	val_X_list = []

	for date in train_X_copy['Last Update']:
		accum_date = ""
		for character in date:
			character = str(character)
			if(character == ' '):
				break
			accum_date += character #Add to only date

		train_X_list.append(accum_date)

	for date in val_X_copy['Last Update']:
		accum_date = ""
		for character in date:
			character = str(character)
			if(character == ' '):
				break
			accum_date += character #Add to only date

		val_X_list.append(accum_date)

	return train_X_list, val_X_list

def PreprocessDateTime(train_X_list, val_X_list):
	list_month_train_X = []
	list_day_train_X = []
	list_year_train_X = []

	list_month_val_X = []
	list_day_val_X = []
	list_year_val_X = []

	pd_train_X = pd.DataFrame()
	pd_val_X = pd.DataFrame()

	#print(train_X_list)
	#preprocess the train_X_list
	counter=1
	try:
		#el train y el val X, cada uno de sus fechas tienen dos formatos diferentes
		#Esta el %m/%d/%y que el ano es 20
		#Y esta el %m/%d/%Y que el ano es 2020

		for date in train_X_list:
			if "/" in date and "2020" in date:
				datetime_object = datetime.strptime(date, '%m/%d/%Y')


			elif "-" in date:
				datetime_object = datetime.strptime(date, '%Y-%m-%d')


			else:
				datetime_object = datetime.strptime(date, '%m/%d/%y')
			counter += 1

			list_month_train_X.append( datetime_object.month )
			list_day_train_X.append(datetime_object.day)
			list_year_train_X.append(datetime_object.year)

		for date in val_X_list:
			if "/" in date and "2020" in date:
				datetime_object = datetime.strptime(date, '%m/%d/%Y')

			elif "-" in date:
				datetime_object = datetime.strptime(date, '%Y-%m-%d')

			else:
				datetime_object = datetime.strptime(date, '%m/%d/%y')

			list_month_val_X.append(datetime_object.month)
			list_day_val_X.append(datetime_object.day)
			list_year_val_X.append(datetime_object.year)

		dictionary_train = {
			"Month": list_month_train_X,
			"Day": list_day_train_X,
			"Year": list_year_train_X
		}

		dictionary_val = {
			"Month": list_month_val_X,
			"Day": list_day_val_X,
			"Year": list_year_val_X
		}

		cols_to_add = ['Month', 'Day', 'Year']
		for colName in cols_to_add:
			pd_train_X[colName] = dictionary_train[colName]
			pd_val_X[colName] = dictionary_val[colName]

	except ValueError:
		raise ValueError("Hubo un error en el indice: " + str(train_X_list[counter-1]) )

		


	return pd_train_X, pd_val_X
	
def ConcatenateDatesToData(train_X_copy, val_X_copy, pd_train_X, pd_val_X):
	pd_train_X.index = train_X_copy.index
	pd_val_X.index = val_X_copy.index

	columns_to_drop = ['Last Update']

	train_X = train_X_copy.drop(columns_to_drop, axis=1)
	val_X = val_X_copy.drop(columns_to_drop, axis=1)

	train_X = pd.concat([train_X, pd_train_X], axis=1)
	val_X = pd.concat([val_X, pd_val_X], axis=1)

	return train_X, val_X

def LabelingEncoder(X):
	#s = (X.dtypes == 'object')
	#object_cols = list(s[s].index)
	#print("\nCategorical Variables: \n" + str(object_cols))
	object_cols = ['Country/Region']
	label_X = X.copy()
	label_encoder = LabelEncoder()

	for col in object_cols:
		label_X[col] = label_encoder.fit_transform(X[col])
	pd_country_region = pd.concat([X['Country/Region'], label_X['Country/Region']], axis=1) #concat two columns and store it into the global variable.
	
	return label_X, pd_country_region

def ImputeYData(y):
	#print("train_y:\n" + str(y))
	y_copy = y.copy()

	#imputer = SimpleImputer()
	#imputed_Y = pd.DataFrame(imputer.fit_transform(y_copy))
	imputed_Y = y_copy.fillna(value=0)

	#imputed_Y.columns = y.columns
	#print( "\nImputed Y: \n" + str(imputed_Y) )
	return imputed_Y

def Score_DataSet(label_X_train, label_X_valid, train_y, val_y, n_estimators):
	#print("\nval Y before training:\n" + str(val_y.head()))
	print("////////////////////////////////////////////////////////////////////////////////////////////")
	print("\nTraining.......")
	model = RandomForestRegressor(random_state=1, criterion="mae", n_estimators=n_estimators)
	model.fit(label_X_train, train_y)
	y_preds = model.predict(label_X_valid)

	mae = mean_absolute_error(val_y, y_preds)
	error_percent = ( (mae * 100)/max(train_y) )
	#100% - 1310
	# x   - mae

	y_preds_pd = pd.DataFrame(y_preds)
	y_preds_pd.columns = ['Predictions']

	#print("\nVal_y after training: ")
	#print(val_y)
	#print("\nConcatenate validation and predictions: ")
	#print(pd.concat([val_y.fillna(value=0), y_preds_pd], axis=1).head())
	print("\nRandom Forest Regressor with: %d n_estimators" %(n_estimators))
	print( "\nMin val_y: " + str(min(val_y)) + "\nMax val_y: " + str(max(val_y)) )
	print("\nMean Absolute Error: " + str(mae))
	print("\nError percent: \n" + str(error_percent))
	#return pd.concat([val_y, y_preds_pd], axis=1)
	return y_preds

entries = os.listdir(r'C:Users\jorge\Desktop\COVID-19 Research\data\COVID-19-master\archived_data\archived_daily_case_updates')
extensions = '.csv'
directory_container = r"C:Users/jorge\Desktop\COVID-19 Research\data\COVID-19-master\archived_data\archived_daily_case_updates"

sorted_data = pd.DataFrame()
for entry in entries:
	if(extensions in entry):
		file_url = directory_container + "/" + entry
		file = pd.read_csv(file_url)
		sorted_data = sorted_data.append(file, ignore_index=True)
		#print("PD File: \n" + str(file))
		#print("sorted data: \n" + str(sorted_data))

#print(sorted_data.columns)
#sorted_data.to_csv(path_or_buf=r'C:\Users\jorge\Desktop\COVID-19 Research\data')

#Beginning of the preprocessing
columns_to_drop = ['Confirmed', 'ConfnSusp', 'Province/State', 'Recovered', 'Notes', 'Deaths', 'Suspected']
#X = sorted_data.drop(columns_to_drop, axis=1) #Drop suspected cases column
y = sorted_data.Confirmed
X = sorted_data.drop(columns_to_drop, axis=1) #Drop suspected cases column

#print("\nNombres de Columnas: \n" + str(X.columns))

label_X, pd_country_region = LabelingEncoder(X)
#print( "\nPandas Country/Region: \n" + str(pd_country_region) )

X = label_X

y = ImputeYData(y) #Y Data contains NAN Data, so i imputed the columns.
#print("\nImputed Y Data: \n" + str(y.head()) )

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)
#print("\ntrain Y Data: \n" + str(train_y.head()) )
#print("\nval Y Data: \n" + str(val_y.head()) )
#print(train_X.dtypes)

train_X_copy = train_X.copy()
val_X_copy = val_X.copy()

train_X_list, val_X_list = FilterDateUntilYear(train_X_copy, val_X_copy)
pd_train_X, pd_val_X = PreprocessDateTime(train_X_list, val_X_list)
#PreprocessDateTime(train_X_list, val_X_list)

train_X, val_X = ConcatenateDatesToData(train_X_copy, val_X_copy, pd_train_X, pd_val_X)#Here is where columns are dropped
#print("\ntrain X columns: \n" + str(train_X.columns ))
#print("\nval X columns: \n" + str(val_X.columns ))
preds = np.full(val_y.shape, None)
for n_estimators in [10, 100, 500]:
	preds = Score_DataSet(train_X, val_X, train_y, val_y, n_estimators)

df_to_csv = pd.DataFrame({"Id":val_y.index,
						  "Confirmed_val":val_y,
						  "Predictions":preds,})
df_to_csv.to_csv(path_or_buf=r'C:\Users\jorge\Desktop\COVID-19 Research\data\preds_validations.csv', index=False)

print("End")
