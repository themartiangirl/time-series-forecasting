import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../data/sales.csv')  # I used dataset from a brand Plantful, feel free to use your own

# Prepare the data for Prophet
df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)


model = Prophet()


model.fit(df)

# future data frame for predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

#plot plot bb
model.plot(forecast)
plt.show()
