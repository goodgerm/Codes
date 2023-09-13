

Risk Classification of Cervical Cancer
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import seaborn as sns

cancer_df = pd.read_csv('/content/risk_factors_cervical_cancer.csv')
cancer_df.head()

len(cancer_df)

cancer_df.keys()

cancer_df.info()

cancer_df = cancer_df.replace('?', np.NaN)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(cancer_df)
scaled_data=scaler.transform(cancer_df)

scaled_data

#PCA
import pandas as pd
from sklearn.decomposition import PCA

# Assuming you have the DataFrame 'cancer_df' with the appropriate data

# Fill NaN values with 0 and convert to integers
scaled_data = cancer_df.fillna(0)

# Create a PCA instance with 2 principal components
pca = PCA(n_components=2)

# Fit and transform the data to the new 2-dimensional space
pca_result = pca.fit_transform(scaled_data)

# Create a new DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA Component 1', 'PCA Component 2'])

print(pca_df)

pca.fit(scaled_data)

scaled_data.shape

x_pca=pca.transform(scaled_data)
x_pca.shape

scaled_data

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer_df['Biopsy'])
plt.xlabel('First principle component')
plt.ylabel('Second Principle component')

cancer_df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],inplace=True,axis=1)

cancer_df = cancer_df.replace('?', np.NaN)

numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']

categorical_df = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS',
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN',
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']

cancer_df = cancer_df.replace('?', np.NaN)
print(cancer_df)

import pandas as pd

for feature in numerical_df:
    print(feature, '', pd.to_numeric(cancer_df[feature], errors='coerce').mean())
    feature_mean = round(pd.to_numeric(cancer_df[feature], errors='coerce').mean(), 1)
    cancer_df[feature] = pd.to_numeric(cancer_df[feature], errors='coerce').fillna(feature_mean)

import pandas as pd

for feature in categorical_df:
    cancer_df[feature] = pd.to_numeric(cancer_df[feature], errors='coerce').fillna(1.0)

category_df = ['Hinselmann', 'Schiller','Citology', 'Biopsy']

import seaborn as sns

for feature in categorical_df:
    sns.countplot(x=feature, data=cancer_df)

import seaborn as sns

for feature in categorical_df:
    sns.catplot(x=feature, data=cancer_df, kind='count')

import seaborn as sns

for category in categorical_df:
    sns.catplot(x=category, y="Hormonal Contraceptives", data=cancer_df, kind='bar', palette="pastel")

import seaborn as sns
g = sns.PairGrid(cancer_df, y_vars=['Hormonal Contraceptives'], x_vars=category_df, aspect=.75, height=3.5)
g.map(sns.barplot, palette="pastel")

cancer_df = cancer_df.replace('?', np.NaN)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

csv_data = '/content/risk_factors_cervical_cancer.csv'
data = pd.read_csv(csv_data)
data = pd.read_csv(csv_data, na_values='?')
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
sns.heatmap(data)
plt.show()

"""SVM classifier"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/content/risk_factors_cervical_cancer.csv")

data = data.replace('?', np.nan)
data = data.dropna()
data = data.astype(float)

x = data.drop(['Biopsy'], axis=1)
y = data["Biopsy"]
corr_matrix=data.corr()
print(corr_matrix)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 1)
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

import matplotlib.pyplot as mtp
mtp.scatter(x_train[:,1],y_train)
mtp.scatter(x_train[:,2],y_train)
mtp.scatter(x_train[:,3],y_train)
mtp.show()

"""Capsule Network"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CapsuleNetwork(keras.Model):
    def __init__(self, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')
        self.primary_caps = layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')
        self.digit_caps = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)  # Add channel dimension as the last dimension
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = layers.Flatten()(x)
        x = self.digit_caps(x)
        return x

num_classes = 8

model = CapsuleNetwork(num_classes)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have x_train, y_train, x_test, and y_test defined and properly formatted

# One-hot encode the labels
y_train_encoded = tf.one_hot(y_train, num_classes)
y_test_encoded = tf.one_hot(y_test, num_classes)

model.fit(x_train, y_train_encoded, batch_size=32, epochs=3, validation_data=(x_test, y_test_encoded))

loss, accuracy = model.evaluate(x_test, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

import numpy as np

# Assuming your input data is stored in a variable called 'x_train' and 'x_test'
# Reshape the input data to add dimensions for batch size and channels
print(x_train.shape)
x_train = np.reshape(x_train, (-1, 35, 1, 1))
x_test = np.reshape(x_test, (-1, 35, 1, 1))
print(x_train.shape)
# Now the shape of x_train and x_test should be (batch_size, height, width, channels)

# Proceed with training and testing your model

"""Neural Network"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers

# Assuming you have x_train, y_train, x_test, and y_test defined and properly formatted

# Flatten the input data to 2D
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalize the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Reshape the scaled data back to the original shape
x_train_scaled = x_train_scaled.reshape(x_train.shape)
x_test_scaled = x_test_scaled.reshape(x_test.shape)

# Define the neural network model
model = keras.Sequential([

    layers.Dense(128, activation='relu'),

    layers.Flatten(input_shape=(x_train.shape[1:])),  # Flatten the input data if needed
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Single output node for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error'])

model.fit(x_train_scaled, y_train, batch_size=32, epochs=3, validation_data=(x_test_scaled, y_test))

# Evaluate the model on the test set
loss,accuracy, mae = model.evaluate(x_test_scaled, y_test)
print("Test Loss:", loss)
print("Accuracy:",accuracy)
print("Test Mean Absolute Error:", mae)