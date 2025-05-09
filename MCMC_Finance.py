# Helper Function to fetch historical stock data
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg' if you want an interactive GUI

import sys
def get_stock_data(ticker, start_date, end_date):
    try:
        # https://medium.com/nerd-for-tech/all-you-need-to-know-about-yfinance-yahoo-finance-library-fa4c6e48f08e
        #data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        return f"Error fetching data: {e}"

ticker_symbol = ["PREIX", "AAPL", "GOOG", "MSFT", "GOLD", "USOIL", "JPM", "BRK-B", "DJIA", "NDX"]  #  T. Rowe Price Equity Index 500 Fund
ticker_index = int(sys.argv[1])
ticker = ticker_symbol[ticker_index]  # Ticker symbol for the stock


start_yr = 2000
end_yr = 2009+ int(sys.argv[2])  # 2020+sys.argv[2] for 2021, 2022, etc.
start_date = f"{start_yr}-01-01"
end_date = f"{end_yr}-01-01"
end2_date = f"{end_yr+1}-01-01"
save_folder = "Figures"
Threshold = 0.40  # 40% percentile



stock_data = get_stock_data(ticker, start_date, end_date)
stock_data_future = get_stock_data(ticker, end_date, end2_date)
stock_data.asfreq('B').index  # set index frequency to business daliy


import pandas as pd

# Assuming you have a DataFrame 'stock_data' with a column 'Close' for stock closing prices
stock_data['Daily Return'] = stock_data['Close'].pct_change()  # percentage return
mean_return = stock_data['Daily Return'].mean()
std_return = stock_data['Daily Return'].std()

# Define thresholds
up_threshold = mean_return + std_return
down_threshold = mean_return - std_return




import matplotlib.pyplot as plt
stock_data['Daily Return'].hist(bins = 100)
plt.axvline(x=up_threshold, color='green', linestyle='--', linewidth=1, label='Up Threshold')
plt.axvline(x=down_threshold, color='red', linestyle='--', linewidth=1, label='Down Threshold')
plt.xlim(-0.05, 0.05)
plt.title(f'Histogram of Daily Returns - {ticker}')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{save_folder}/histogram_{ticker}.png')
plt.close()




def classify_state(return_value, up_thresh, down_thresh):
    if return_value > up_thresh:
        return 'Up'
    elif return_value < down_thresh:
        return 'Down'
    else:
        return 'Stagnant'

stock_data['Markov State'] = stock_data['Daily Return'].apply(lambda x: classify_state(x, up_threshold, down_threshold))
# Shift 'Markov State' column to get previous state
stock_data['Prev State'] = stock_data['Markov State'].shift(1)

# Create a transition matrix and Normalize to get probabilities
transition_counts = pd.crosstab(stock_data['Prev State'], stock_data['Markov State'])
transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)





#Find equilibrium matrix
import numpy as np

def find_equilibrium(transition_matrix):
    # Add a small number to ensure the matrix is not singular
    transition_matrix = transition_matrix + 1e-6
    # Initialize the equilibrium vector
    equilibrium = np.random.rand(transition_matrix.shape[0])
    # Normalize the equilibrium vector
    equilibrium = equilibrium / np.sum(equilibrium)
    # Iterate until convergence
    # print(equilibrium, transition_matrix)
    count = 0
    while True:
        new_equilibrium = np.dot(equilibrium, transition_matrix)
        # Check for convergence
        if np.allclose(equilibrium, new_equilibrium):
            break
        # Update the equilibrium vector
        equilibrium = new_equilibrium
        count += 1
    return equilibrium, count

# Calculate transition matrix

equilibrium, count = find_equilibrium(transition_matrix)
print(equilibrium, count)
states = ['Down', 'Stagnant', 'Up']



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# Fit the Student's t distribution to the returns data
params = t.fit(stock_data['Daily Return'].dropna())

# Get the parameters of the distribution
df_t, mu, sigma = params

# from above
print("Transition Matrix:\n", transition_matrix)
initial_price = stock_data['Close'].iloc[-1]  # Initial stock price

# 4. Monte Carlo Simulation based on Markov Chain
n_simulations = 1000  # Number of simulation paths
n_days = 252  # Simulate for 1 year (252 trading days)

# Initial state probabilities
initial_state = stock_data['Markov State'].iloc[-1]
initial_state_probabilities = [1 if state == initial_state else 0 for state in states]

# Simulating the paths
import tqdm
from tqdm import tqdm
simulated_paths = []
for sim in tqdm(range(n_simulations), desc="Running simulations"):
    simulated_returns = []
    current_state = np.random.choice(states, p=initial_state_probabilities)

    for day in range(n_days):
        # Simulate return based on the current state using Student's t-distribution
        if current_state == "Up":
            daily_return = t.rvs(df_t, loc=mu + sigma, scale=sigma)
        elif current_state == "Down":
            daily_return = t.rvs(df_t, loc=mu - sigma, scale=sigma)
        else:
            daily_return = t.rvs(df_t, loc=mu, scale=sigma)

        simulated_returns.append(daily_return)
        # Transition to the next state based on the transition matrix
        current_state = np.random.choice(states, p=transition_matrix.loc[current_state].values)

    # Cumulative price based on returns
    simulated_prices = initial_price.iloc[0] * np.exp(np.cumsum(simulated_returns))
    simulated_paths.append(simulated_prices)

# Convert the list of paths to a DataFrame
simulated_paths_df = pd.DataFrame(simulated_paths).T






# 5. Plotting the Monte Carlo simulation
plt.figure(figsize=(10, 6))
plt.plot(simulated_paths_df, color="lightblue", alpha=0.1)

# Add 5% and 95% percentile lines
percentiles_5 = simulated_paths_df.quantile(0.05, axis=1)
percentiles_95 = simulated_paths_df.quantile(0.95, axis=1)
percentiles_40 = simulated_paths_df.quantile(Threshold, axis=1)

plt.plot(percentiles_5, color="blue", linestyle="--", label="5th Percentile")
plt.plot(percentiles_40, color="black", linestyle="--", label=f"{int(100*Threshold)}th Percentile")
plt.plot(percentiles_95, color="green", linestyle="--", label="95th Percentile")
days = range(len(stock_data_future))
plt.plot(days, stock_data_future["Close"], color="red", label="Actual price")


last_price = stock_data['Close'].iloc[-1].iloc[0]
plt.plot(days, last_price * np.ones(len(days)), color="orange", label="Last Price")

# Add labels and title
plt.title(f"MCMC of Stock Close Price ({ticker}), {end_yr}-{end_yr+1}")
plt.xlabel("Days")
plt.ylabel("Price")
plt.ylim(min(percentiles_5), max(percentiles_95))
plt.legend()
# plt.show()
plt.savefig(f'{save_folder}/monte_carlo_simulation_{ticker}_{end_yr+1}.png')
plt.close()


# 5. Calculate the predicted and actual growth percentages
Predicted_price = percentiles_40.iloc[-1]
Actual_price = stock_data_future['Close'].iloc[-1][ticker] # 40th percentile price at the end of the simulation period
Growth_percent_predicted = (Predicted_price - last_price) / last_price * 100
Growth_percent_actual = (Actual_price - last_price) / last_price * 100
print(f"Predicted Growth Percentage: {Growth_percent_predicted:.2f}%") 
print(f"Actual Growth Percentage: {Growth_percent_actual:.2f}%")




# 6. Saving in pickle file
# Create a dictionary to store all the variables
import pickle
cell_info = {
    "Actual_price": Actual_price,
    "Growth_percent_actual": Growth_percent_actual,
    "Growth_percent_predicted": Growth_percent_predicted,
    "Predicted_price": Predicted_price,
    "last_price": last_price}

# Save the dictionary to a pickle file
with open(f"{save_folder}/Prediction_{ticker}_{end_yr+1}.pkl", "wb") as f:
    pickle.dump(cell_info, f)