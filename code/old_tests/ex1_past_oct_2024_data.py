import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier


""" --- 1. Fetch FTSE Data from Yahoo Finance --- """

# Fetch FTSE 100 data
ftse_ticker = "^FTSE"  # FTSE 100 Index
ftse_data = yf.download(ftse_ticker, start="2014-04-30", end="2024-11-30", interval="1mo")

# Add derived features
ftse_data['Date'] = ftse_data.index
ftse_data['Return'] = np.log(ftse_data['Adj Close'] / ftse_data['Adj Close'].shift(1))
ftse_data['6M_Return'] = ftse_data['Return'].rolling(window=6).sum()
ftse_data['Volatility'] = ftse_data['Return'].rolling(6).std()
ftse_data['Moving_Avg'] = ftse_data['Return'].rolling(6).mean()
# Define momentum as the difference between recent log return and its moving average
ftse_data['Momentum'] = ftse_data['Return'] - ftse_data['Moving_Avg']

# Keep only data after 2024-12-01
ftse_data = ftse_data[ftse_data['Date'] >= '2014-12-01']
ftse_data.reset_index(drop=True, inplace=True)


""" --- 2. Load Risk-Free Rate Data --- """

# Load and clean risk-free rate data from the Excel file
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the Excel file
file_path = os.path.join(script_dir, "cw2024AP.xlsx")

risk_free_data = pd.read_excel(file_path, sheet_name="Sheet1", skiprows=3)
risk_free_data = risk_free_data[["Unnamed: 0", "Risk Free Asset"]]
risk_free_data.columns = ["Date", "Risk_Free_Asset"]
risk_free_data["Date"] = pd.to_datetime(risk_free_data["Date"], errors="coerce")
risk_free_data["Risk_Free_Asset"] = pd.to_numeric(risk_free_data["Risk_Free_Asset"], errors="coerce")

# Define the start and end dates
start_date = "2014-06-30"
end_date = "2024-10-31"

# Filter the risk-free rate data for the specified period
risk_free_data = risk_free_data[(risk_free_data["Date"] >= start_date) & (risk_free_data["Date"] <= end_date)]
risk_free_data["Risk_Free_Rate"] = risk_free_data["Risk_Free_Asset"] / 100

risk_free_data["6M_Cumulative_Risk_Free_Rate"] = (
    np.log(1 + risk_free_data["Risk_Free_Rate"]).rolling(window=6).sum()
)

#risk_free_data = risk_free_data.dropna()
risk_free_data.reset_index(drop=True, inplace=True)
risk_free_data['Date'] = ftse_data['Date']


""" --- 3. Merge Datasets --- """

# Merge FTSE data with risk-free data
merged_data = pd.merge(ftse_data, risk_free_data, on="Date", how="inner")
merged_data = merged_data.dropna()
merged_data.reset_index(drop=True, inplace=True)

# Define target: 1 if market return > risk-free rate
merged_data["Target"] = (merged_data["6M_Return"] > merged_data["6M_Cumulative_Risk_Free_Rate"]).astype(int)


""" --- 4. Define Features and Models --- """

# Define features and target
features = ["Volatility", "Momentum", "Moving_Avg"]
X = merged_data[features]
y = merged_data["Target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define machine learning models
models = {
	"Random Forest": RandomForestClassifier(random_state=42),
	"Gradient Boosting": GradientBoostingClassifier(random_state=42),
	"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
	"SVM": SVC(probability=True, random_state=42),
	"k-NN": KNeighborsClassifier(),
	"Neural Network": MLPClassifier(max_iter=1000, random_state=42),
}

# Define hyperparameter grids for each model
param_grids = {
	"Random Forest": {
		"n_estimators": [50, 100, 200],
		"max_depth": [None, 10, 20, 30],
		"min_samples_split": [2, 5, 10],
		"min_samples_leaf": [1, 2, 4]
	},
	"Gradient Boosting": {
		"n_estimators": [50, 100, 200],
		"learning_rate": [0.01, 0.1, 0.2],
		"max_depth": [3, 5, 7],
		"subsample": [0.8, 1.0]
	},
	"Logistic Regression": {
		"C": [0.01, 0.1, 1, 10],
		"penalty": ["l2"],
		"solver": ["lbfgs"]
	},
	"SVM": {
		"C": [0.01, 0.1, 1, 10],
		"kernel": ["linear", "rbf", "poly"],
		"gamma": ["scale", "auto"],
		'max_iter':[5000,10000,50000]
	},
	"k-NN": {
		"n_neighbors": [3, 5, 7, 10],
		"weights": ["uniform", "distance"],
		"metric": ["minkowski", "euclidean", "manhattan"]
	},
	"Neural Network": {
		"hidden_layer_sizes": [(50,), (100,), (50, 50)],
		"activation": ["relu", "tanh"],
		"solver": ["adam", "sgd"],
		"alpha": [0.0001, 0.001, 0.01]
	}
}


def calculate_sharpe_ratio(returns, risk_free_rate):
	"""
	Calculate the Sharpe Ratio for a given series of returns.
	
	Parameters:
	- returns (pd.Series): Strategy or asset returns.
	- risk_free_rate (float): Daily risk-free rate.

	Returns:
	- sharpe_ratio (float): The Sharpe Ratio.
	"""
	excess_returns = returns - risk_free_rate
	avg_excess_return = excess_returns.mean()
	std_excess_return = excess_returns.std()
	return avg_excess_return / std_excess_return if std_excess_return != 0 else 0


""" --- 5. Train and Evaluate Models --- """

# Set up logging to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store results
results = {}
performance_metrics = []

for name, model in models.items():
	logging.info(f"Starting hyperparameter tuning for {name}...")

	# Perform grid search for hyperparameter tuning
	param_grid = param_grids[name]
	grid_search = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		scoring="accuracy",
		cv=3,
		n_jobs=-1,
		verbose=1  # Verbose outputs progress of GridSearchCV
	)

	logging.info(f"Grid Search for {name} with parameters: {param_grid}")
	grid_search.fit(X_train, y_train)

	# Get the best model and its parameters
	best_model = grid_search.best_estimator_
	best_params = grid_search.best_params_
	logging.info(f"Best parameters for {name}: {best_params}")

	# Evaluate the best model
	y_pred = best_model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	# Calculate strategy returns based on predictions
	strategy_returns = np.where(
		best_model.predict(X) == 1,
		merged_data["6M_Return"],
		merged_data["6M_Cumulative_Risk_Free_Rate"]
	)

	# Calculate Sharpe Ratio
	sharpe_ratio = calculate_sharpe_ratio(strategy_returns, merged_data["6M_Cumulative_Risk_Free_Rate"])
	
	predictions = best_model.predict(X)
	# Store results
	results[name] = {
		"model": best_model,
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_score": f1,
		"sharpe_ratio": sharpe_ratio,
		"predictions": predictions,
		"best_params": best_params
	}

	# Append metrics to the performance table
	performance_metrics.append({
		"Model": name,
		"Accuracy": accuracy,
		"Precision": precision,
		"Recall": recall,
		"F1-Score": f1,
		"Sharpe Ratio": sharpe_ratio
	})
	logging.info(f"Finished training {name} with Sharpe Ratio: {sharpe_ratio:.2f}")

	monthly_decisions = []

	# Generate a list of dictionaries with months and decisions
	monthly_decisions = [
		{"Month": month.to_period("M"), "Decision": "Invest" if decision == 1 else "Don't Invest"}
		for month, decision in zip(merged_data['Date'], predictions)
	]

	# Print the monthly decisions
	for decision in monthly_decisions:
		print(f"{decision['Month']}: {decision['Decision']}")


# Create and display performance summary
performance_df = pd.DataFrame(performance_metrics)
logging.info("\nModel Performance Summary:\n")
logging.info(performance_df)

# Sort models by Sharpe Ratio and select the top 3
top_3_models = performance_df.sort_values(by="Sharpe Ratio", ascending=False).head(3)["Model"].tolist()
logging.info(f"Top 3 models selected for stacking: {top_3_models}")


# Define base models for stacking
stacking_estimators = [(name, results[name]["model"]) for name in top_3_models]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create Stacking Ensemble
stacking_clf = StackingClassifier(
	estimators=stacking_estimators,
	final_estimator=meta_model,
	cv=3
)

# Train Stacking Ensemble
stacking_clf.fit(X_train, y_train)

# Evaluate Stacking Ensemble
y_pred_stack = stacking_clf.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
precision_stack = precision_score(y_test, y_pred_stack)
recall_stack = recall_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)

# Calculate strategy returns for Stacking Ensemble
stacking_strategy_returns = np.where(
	stacking_clf.predict(X) == 1,
	merged_data["6M_Return"],
	merged_data["6M_Cumulative_Risk_Free_Rate"]
)

# Calculate Sharpe Ratio for Stacking Ensemble
sharpe_stack = calculate_sharpe_ratio(stacking_strategy_returns, merged_data["6M_Cumulative_Risk_Free_Rate"])

# Add Stacking Ensemble results to performance metrics
performance_metrics.append({
	"Model": "Stacking Ensemble",
	"Accuracy": accuracy_stack,
	"Precision": precision_stack,
	"Recall": recall_stack,
	"F1-Score": f1_stack,
	"Sharpe Ratio": sharpe_stack
})
logging.info(f"Stacking Ensemble: Accuracy={accuracy_stack:.2f}, Precision={precision_stack:.2f}, Recall={recall_stack:.2f}, F1-Score={f1_stack:.2f}, Sharpe Ratio={sharpe_stack:.2f}")

# Generate monthly investment decisions for Stacking Ensemble
stacking_predictions = stacking_clf.predict(X)

monthly_decisions_stacking_strategy = []

# Generate a list of dictionaries with months and decisions
monthly_decisions_stacking_strategy = [
	{"Month": month.to_period("M"), "Decision": "Invest" if decision == 1 else "Don't Invest"}
	for month, decision in zip(merged_data['Date'], stacking_predictions)
]

# Print monthly investment decisions for stacking ensemble
print("\nMonthly Investment Decisions (Stacking Ensemble):")
for decision in monthly_decisions_stacking_strategy:
	print(f"{decision['Month']}: {decision['Decision']}")


""" --- 6. Calculate Strategy Returns --- """

# Initialize cumulative returns for each model
for name, result in results.items():
	merged_data[f"{name}_Strategy_Return"] = np.where(
		result["predictions"] == 1,
		merged_data["6M_Return"],
		merged_data["6M_Cumulative_Risk_Free_Rate"]
	)

	# Calculate cumulative returns using summation for log returns
	merged_data[f"{name}_Cumulative"] = (
		merged_data[f"{name}_Strategy_Return"].cumsum()
	)

# Add stacking strategy returns to merged_data
merged_data["Stacking_Strategy_Return"] = stacking_strategy_returns
merged_data["Stacking_Cumulative"] = merged_data["Stacking_Strategy_Return"].cumsum()


""" --- 7. Plot Results --- """

# Plot cumulative returns of all models and the market
plt.figure(figsize=(14, 8))

for name in results.keys():
	plt.plot(
		merged_data["Date"],
		merged_data[f"{name}_Cumulative"],
		label=f"{name} Strategy"
	)

# Plot Stacking Ensemble results
plt.plot(
	merged_data["Date"],
	merged_data["Stacking_Cumulative"],
	label="Stacking Ensemble Strategy",
)

plt.plot(
	merged_data["Date"],
	merged_data["6M_Return"],
	label="Market Returns",
	linewidth=2,
	linestyle="--"
)

# Plot Risk-Free cumulative returns
plt.plot(
    merged_data["Date"],
    merged_data["6M_Cumulative_Risk_Free_Rate"],
    label="Risk-Free Returns",
    linestyle="-.",
    linewidth=1.5
)

plt.legend()
plt.title("Cumulative Returns of Different Models")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid()
plt.show()
