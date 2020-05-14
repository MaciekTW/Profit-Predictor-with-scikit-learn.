import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


dataset = pd.read_csv( 'Info.csv' )
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer( transformers=[ ( 'encoder', OneHotEncoder(), [3] ) ], remainder='passthrough' )
X = np.array( ct.fit_transform(X) )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 1/5, random_state = 0 )

regressor = LinearRegression()
regressor.fit( X_train, y_train )

y_predict = regressor.predict( X_test )
predictions = np.concatenate( ( y_predict.reshape( len(y_predict),1 ), y_test.reshape( len(y_test), 1 ) ), 1 )


for item,item2 in predictions:
    deviation = abs((item2-item)/item) * 100
    print( "Prediction: {0} Exact value: {1} Deviation = {2}% ".format( item2, round(item,2), round(deviation,2) ) )