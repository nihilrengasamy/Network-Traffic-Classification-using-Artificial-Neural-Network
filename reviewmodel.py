import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Importing Dataset
data = pd.read_csv("D:\mini\ARFF Files\Scenario A1\TimeBasedFeatures-Dataset-15s-VPN.csv")
print(data)

# datatype of columns and missing values is nill
print(data.info())

# Converting the columns of dtype int to float
data[['duration','total_fiat','total_biat','min_fiat','min_biat','min_flowiat','max_flowiat','min_active','max_active','min_idle','max_idle']] = data[['duration','total_fiat','total_biat','min_fiat','min_biat','min_flowiat','max_flowiat','min_active','max_active','min_idle','max_idle']].astype('float')

print(data.info())

print(data['class1'].value_counts())

#Label Encoding Categorical Column
#Label Encoding
from sklearn import preprocessing 
label = preprocessing.LabelEncoder() 
 
data['class1']= label.fit_transform(data['class1'])
print(data['class1'].value_counts())


# Creating the Correlation matrix and Selecting the Upper trigular matrix

cor_matrix = data.corr().abs()
print(cor_matrix)

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(to_drop)


data = data[["duration","total_fiat","total_biat","min_fiat","min_biat","max_fiat","mean_biat","flowPktsPerSecond","flowBytesPerSecond","min_flowiat","mean_flowiat","std_flowiat","std_active","class1"]]
data

#Train Test Split 
X = pd.DataFrame(data.iloc[:, 0:13].values)
y = data.iloc[:, 13].values




# Split the X and Y Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

# Perform Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Model Bulding
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the Artificial Neural Network
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(units=13, activation='relu', kernel_initializer='uniform', input_dim = (13)))

# Add the second hidden layer
classifier.add(Dense(units = 64 , kernel_initializer = 'uniform' , activation = 'relu'))

# Add the output layer
classifier.add(Dense(kernel_initializer = 'uniform',activation = 'sigmoid',units = 1))

# Compile the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 25)

print(classifier.summary())



from keras.models import load_model
classifier.save('classifier.h5')
model_final = load_model('classifier.h5')

X_test[0].shape

model_final.summary()

X_1 = X_test[0].reshape((1,13))

print(model_final.predict(X_1))


