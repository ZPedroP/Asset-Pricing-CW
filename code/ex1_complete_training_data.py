import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier

# ---------------- CONFIGURATION ---------------- #
CONFIG = {
	"ftse_ticker": "^FTSE",  # Ticker for the FTSE index

	"ftse_start_date": "1995-12-31",  # Start date for historical data
	"ftse_end_date": "2024-11-30",  # End date for historical data
	"interval": "1mo",  # Data interval (monthly in this case)

	"risk_free_file": "cw2024AP.xlsx",  # File containing risk-free rate data
	"risk_free_sheet": "Sheet1",  # Sheet name in the Excel file
	"risk_free_skiprows": 3,  # Number of rows to skip in the Excel file

	"train_cutoff_date": "2014-10-31",  # Date separating training and test data
	"test_start_date": "2014-11-01",  # Start date for test data

	"features": ["Volatility", "Momentum", "Moving_Avg"],  # Features used for modeling

	"model_parameters": {
		"Random Forest": {
			"n_estimators": [50, 100, 200],
			"max_depth": [None, 10, 20, 30],
			"min_samples_split": [2, 5, 10],
			"min_samples_leaf": [1, 2, 4]
		},  # Parameter grid for Random Forest
		"Gradient Boosting": {
			"n_estimators": [50, 100, 200],
			"learning_rate": [0.01, 0.1, 0.2],
			"max_depth": [3, 5, 7],
			"subsample": [0.8, 1.0]
		},  # Parameter grid for Gradient Boosting
		"Logistic Regression": {
			"C": [0.01, 0.1, 1, 10],
			"penalty": ["l2"],
			"solver": ["lbfgs"]
		},  # Parameter grid for Logistic Regression
		"SVM": {
			"C": [0.01, 0.1, 1, 10],
			"kernel": ["linear", "rbf", "poly"],
			"gamma": ["scale", "auto"],
			'max_iter':[5000,10000,50000]
		},  # Parameter grid for Support Vector Machines
		"k-NN": {
			"n_neighbors": [3, 5, 7, 10],
			"weights": ["uniform", "distance"],
			"metric": ["minkowski", "euclidean", "manhattan"]
		},  # Parameter grid for k-Nearest Neighbors
		"Neural Network": {
			"hidden_layer_sizes": [(50,), (100,), (50, 50)],
			"activation": ["relu", "tanh"],
			"solver": ["adam", "sgd"],
			"alpha": [0.0001, 0.001, 0.01]
		}  # Parameter grid for Neural Networks
	}
}

config = CONFIG

""" --- 1. Fetch FTSE Data from Yahoo Finance --- """

def fetch_market_data(ticker="^FTSE", start_date="1995-12-31", end_date="2024-11-30", interval="1mo"):
	"""
	Fetch market data from Yahoo Finance and compute derived features.
	
	Args:
		ticker (str): The stock ticker symbol.
		start_date (str): The start date for historical data.
		end_date (str): The end date for historical data.
		interval (str): Data interval (e.g., '1mo' for monthly).

	Returns:
		pd.DataFrame: Processed market data with derived features.
	"""
	# Fetch data using yfinance
	data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

	# Add derived features
	data['Date'] = data.index
	data['Return'] = (data['Adj Close'] / data['Adj Close'].shift(1)) - 1  # Monthly returns
	data['6M_Return'] = data['Return'].rolling(window=6).sum()  # 6-month cumulative return
	data['Volatility'] = data['Return'].rolling(6).std()  # Rolling standard deviation (volatility)
	data['Moving_Avg'] = data['Return'].rolling(6).mean()  # Rolling mean (moving average)
	data['Momentum'] = data['Return'] - data['Moving_Avg']  # Momentum indicator
	data.reset_index(drop=True, inplace=True)

	return data


""" --- 2. Load Risk-Free Rate Data --- """

def load_risk_free_data(filename="cw2024AP.xlsx", sheet_name="Sheet1", skiprows=3, start_date="1995-12-31", end_date="2024-10-31"):
	"""
	Load risk-free rate data and preprocess it.
	
	Args:
		filename (str): Filename of the Excel file.
		sheet_name (str): Name of the sheet containing the data.
		skiprows (int): Number of rows to skip in the sheet.
		start_date (str): Start date for filtering data.
		end_date (str): End date for filtering data.

	Returns:
		pd.DataFrame: Processed risk-free rate data.
	"""
	# Load risk-free rate data
	script_dir = os.path.dirname(os.path.abspath(__file__))
	filepath = os.path.join(script_dir, filename)
	data = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=skiprows)
	data = data[["Unnamed: 0", "Risk Free Asset"]]
	data.columns = ["Date", "Risk_Free_Asset"]
	data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
	data["Risk_Free_Asset"] = pd.to_numeric(data["Risk_Free_Asset"], errors="coerce")

	# Calculate risk-free rate as a percentage
	data["Risk_Free_Rate"] = data["Risk_Free_Asset"] / 100
	data["6M_Cumulative_Risk_Free_Rate"] = data["Risk_Free_Rate"] / 2  # Semi-annual rate

	# Filter data based on the provided date range
	data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
	data.reset_index(drop=True, inplace=True)
	
	return data


""" --- 3. Merge Datasets --- """

def merge_datasets(market_data, risk_free_data):
	"""
	Merge market data with risk-free rate data.
	
	Args:
		market_data (pd.DataFrame): Market data with derived features.
		risk_free_data (pd.DataFrame): Risk-free rate data.

	Returns:
		pd.DataFrame: Combined dataset with target labels.
	"""
	# Merge data on the 'Date' column
	risk_free_data['Date'] = market_data['Date']
	merged_data = pd.merge(market_data, risk_free_data, on="Date", how="inner")
	merged_data = merged_data.dropna()
	merged_data.reset_index(drop=True, inplace=True)

	# Define target variable: 1 if the market return exceeds risk-free rate, else 0
	merged_data["Target"] = (merged_data["6M_Return"] > merged_data["6M_Cumulative_Risk_Free_Rate"]).astype(int)

	return merged_data


""" --- 4. Define Features and Models --- """

ftse_data = fetch_market_data()
risk_free_data = load_risk_free_data()
merged_data = merge_datasets(ftse_data, risk_free_data)

# Split data into training and testing sets
train_data = merged_data[merged_data['Date'] <= config["train_cutoff_date"]]
test_data = merged_data[merged_data['Date'] >= config["test_start_date"]]
X_train, y_train = train_data[config["features"]], train_data["Target"]
X_test, y_test = test_data[config["features"]], test_data["Target"]

# Define models
models = {
	"Random Forest": RandomForestClassifier(random_state=42),
	"Gradient Boosting": GradientBoostingClassifier(random_state=42),
	"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
	"SVM": SVC(probability=True, random_state=42),
	"k-NN": KNeighborsClassifier(),
	"Neural Network": MLPClassifier(max_iter=2500, random_state=42),
}


""" --- 5. Train and Evaluate Models --- """

def calculate_sharpe_ratio(returns, risk_free_rate):
	"""
	Calculate the Sharpe Ratio for a given series of returns.
	
	Args:
		returns (np.array): Series of portfolio returns.
		risk_free_rate (np.array): Series of risk-free rates.

	Returns:
		float: Sharpe ratio.
	"""
	excess_returns = returns - risk_free_rate
	avg_excess_return = excess_returns.mean()
	std_excess_return = excess_returns.std()
	return avg_excess_return / std_excess_return if std_excess_return != 0 else 0

def train_models(models, param_grids, X_train, y_train):
	"""
	Train and tune models using GridSearchCV.

	Args:
		models (dict): Dictionary of models to train.
		param_grids (dict): Dictionary of parameter grids for each model.
		X_train (pd.DataFrame): Training feature set.
		y_train (pd.Series): Training labels.

	Returns:
		dict: Dictionary containing trained models and their best parameters.
	"""
	best_params = {}
	for name, model in models.items():
		# Perform grid search with cross-validation
		grid_search = GridSearchCV(
			estimator=model,
			param_grid=param_grids[name],
			scoring="accuracy",
			cv=3,
			n_jobs=-1,
			verbose=1
		)
		grid_search.fit(X_train, y_train)
		best_params[name] = {
			"model": grid_search.best_estimator_,
			"best_params": grid_search.best_params_
		}
	return best_params

def evaluate_model(model, X_test, y_test, test_data):
	"""
	Evaluate the model on test data.

	Args:
		model: Trained model.
		X_test (pd.DataFrame): Test feature set.
		y_test (pd.Series): Test labels.
		test_data (pd.DataFrame): Test dataset including returns.

	Returns:
		dict: Evaluation metrics and predictions.
	"""
	# Predict test labels
	y_pred = model.predict(X_test)

	# Calculate evaluation metrics
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	# Calculate Sharpe ratio
	sharpe_ratio = calculate_sharpe_ratio(
		np.where(y_pred == 1, test_data["6M_Return"], test_data["6M_Cumulative_Risk_Free_Rate"]),
		test_data["6M_Cumulative_Risk_Free_Rate"]
	)
	
	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_score": f1,
		"sharpe_ratio": sharpe_ratio,
		"predictions": y_pred
	}

results = train_models(models, config['model_parameters'], X_train, y_train)

for name, result in results.items():
	evaluation = evaluate_model(result["model"], X_test, y_test, test_data)
	results[name].update(evaluation)

def train_ensemble_model(trained_models, X_train, y_train):
	"""
	Train a stacking ensemble model using the trained base models.

	Args:
		trained_models (dict): Dictionary of trained base models.
		X_train (pd.DataFrame): Training feature set.
		y_train (pd.Series): Training labels.

	Returns:
		StackingClassifier: Trained stacking ensemble model.
	"""
	# Sort the dictionary items by Sharpe Ratio in descending order
	sorted_models = sorted(trained_models.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True)

	# Extract the top 3 models
	best_performing_models = [model[0] for model in sorted_models[:3]]

	# Define base models for stacking
	estimators = [(name, trained_models[name]["model"]) for name in best_performing_models]

	ensemble = StackingClassifier(
		estimators=estimators,
		final_estimator=LogisticRegression(max_iter=1000, random_state=42)
	)
	ensemble.fit(X_train, y_train)

	return ensemble

results_ensemble = train_ensemble_model(results, X_train, y_train)
evaluation_ensemble = evaluate_model(results_ensemble, X_test, y_test, test_data)
results["Stacking Ensemble"] = evaluation_ensemble


""" --- 6. Calculate Strategy Returns --- """

# Calculate log returns for each model
for name, result in results.items():
    # Compute strategy log returns
    test_data[f"{name}_Log_Return"] = np.where(
        result["predictions"] == 1,
        np.log1p(test_data["6M_Return"]),  # Use log(1 + 6M market return)
        np.log1p(test_data["6M_Cumulative_Risk_Free_Rate"])  # Use log(1 + 6M risk-free return)
    )

    # Compute cumulative log returns
    test_data[f"{name}_Cumulative_Log_Return"] = test_data[f"{name}_Log_Return"].cumsum()

# Calculate log returns for Stacking Strategy
test_data["Stacking_Log_Return"] = np.where(
    results['Stacking Ensemble']['predictions'] == 1,
    np.log1p(test_data["6M_Return"]),  # Use log(1 + 6M market return)
    np.log1p(test_data["6M_Cumulative_Risk_Free_Rate"])  # Use log(1 + 6M risk-free return)
)
# Compute cumulative log returns for the stacking strategy
test_data["Stacking_Cumulative_Log_Return"] = test_data["Stacking_Log_Return"].cumsum()

# Assuming 'results' dictionary contains predictions for each model
for name, result in results.items():
	test_data[f"{name}_Predictions"] = results[name]["predictions"]

test_data["Stacking_Ensemble_Predictions"] = results['Stacking Ensemble']['predictions']

# Filter rows for every 6 months from the first month
six_months_intervals = test_data.iloc[::6].reset_index(drop=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file path for saving predictions
if not os.path.exists(script_dir):
    os.makedirs(script_dir)

output_file = os.path.join(script_dir, "model_predictions.xlsx")

# Convert the Date column to a string format explicitly before saving
six_months_intervals["Date"] = six_months_intervals["Date"].dt.strftime("%Y-%m-%d")

# Create a DataFrame to hold predictions from all models
predictions_df = six_months_intervals[["Date"]].copy()

# Add predictions from each model to the DataFrame
for name in results.keys():
    predictions_df[name] = six_months_intervals[f"{name}_Predictions"]

predictions_df["Stacking Ensemble"] = six_months_intervals["Stacking_Ensemble_Predictions"]

# Save the predictions DataFrame to an Excel file
predictions_df.to_excel(output_file, index=False)

print(f"Model Predictions saved to {output_file}")


""" --- 7. Plot Results --- """

# Plot Cumulative log returns of all models and the market
plt.figure(figsize=(14, 8))

for name in results.keys():
    plt.plot(
        test_data["Date"],
        test_data[f"{name}_Cumulative_Log_Return"],
        label=f"{name} Strategy"
    )

# Plot Stacking Ensemble results
plt.plot(
    test_data["Date"],
    test_data["Stacking_Cumulative_Log_Return"],
    label="Stacking Ensemble Strategy",
)

# Plot cumulative log market returns
test_data["Market_Log_Return"] = np.log1p(test_data["6M_Return"])
test_data["Market_Cumulative_Log_Return"] = test_data["Market_Log_Return"].cumsum()

plt.plot(
    test_data["Date"],
    test_data["Market_Cumulative_Log_Return"],
    label="Market Returns",
    linestyle="--",
    linewidth=2
)

# Plot cumulative log risk-free returns
test_data["Risk_Free_Log_Return"] = np.log1p(test_data["6M_Cumulative_Risk_Free_Rate"])
test_data["Risk_Free_Cumulative_Log_Return"] = test_data["Risk_Free_Log_Return"].cumsum()

plt.plot(
    test_data["Date"],
    test_data["Risk_Free_Cumulative_Log_Return"],
    label="Risk-Free Rate Returns",
    linestyle="-.",
    linewidth=1.5
)

plt.legend()
plt.title("Comparison between the Cumulative Log Returns of Different Market Prediction Models")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Returns")
plt.grid()
plt.savefig('Figure_X.png', format='png', dpi=1200)
plt.show()
