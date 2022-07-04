# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN

# %%
dataName = "17082017_08012022_1DAY.csv"

data = pd.read_csv(dataName) 
data = data.drop(columns="Unnamed: 0")
data.head()

# %% Dropping all columns except "Open"
arr_Open = data.Open
arr_Open.head()

# %% Converting arr_Open array to df
df_Open = pd.DataFrame(arr_Open)
df_Open.head()

# %% Feature Scaling applying MinMaxScaler methodology
scaler = MinMaxScaler()
df = scaler.fit_transform(df_Open)
df

# %% Creating a data structure with 90 timesteps and 1 output
df_X, df_Y = [],[]

first = 10     # first array = time steps
last = len(df) # last array

for i in range(first, last):
    df_X.append(df[i - first : i])
    df_Y.append(df[i])

# %% Converting array to np
df_X, df_Y = np.array(df_X), np.array(df_Y) 

# %% Splitting Dataset

test_days = 30; 

x_train = df_X[:-test_days] # the days before the last 30 days
y_train = df_Y[:-test_days]

x_test = df_X[len(df_X)-test_days:] # the last 30 days
y_test = df_Y[len(df_X)-test_days:]

# %%
def shape_():
    print("x_train shape:",x_train.shape, "\n x_test shape:",x_test.shape)
    print("y_train shape:",y_train.shape, "\n y_test shape:",y_test.shape)

shape_()

# %% Creating RNN Model

# Initialising the RNN
regressor = Sequential()
# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 10,activation='tanh', return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 10,activation='relu', return_sequences = True))
#regressor.add(Dropout(0.2))
# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 2,activation='tanh', return_sequences = True))
#regressor.add(Dropout(0.2))
# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 20))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))

# %%
regressor.summary()

# %%
epochs, batch_size = 60, 15
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

# %%
y_train_pred = regressor.predict(x_train)
y_train_pred = scaler.inverse_transform(y_train_pred)

y_train = scaler.inverse_transform(y_train)

y_test_pred = regressor.predict(x_test)
y_test_pred = scaler.inverse_transform(y_test_pred)

y_test = scaler.inverse_transform(y_test)

# %% Visualising the results
import matplotlib.pyplot as plt

def ploty(real, predicted):
    plt.plot(real, color = 'red', label = '  Real Price')
    plt.plot(predicted, color = 'blue', label = '  Predicted Price')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_():
    ploty(y_train, y_train_pred) # TRAIN
    ploty(y_test, y_test_pred) # TEST

# %%
plot_()

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("------------------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")
def evals_():
    print("y_train / y_train_pred ")
    eval_metrics(y_train, y_train_pred)
    print("y_test / y_test_pred ")
    eval_metrics(y_test, y_test_pred)
# %%
evals_()
# %%
