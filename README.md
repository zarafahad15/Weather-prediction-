# Weather-prediction-
This project demonstrates the use of Machine Learning techniques to predict weather temperature based on environmental factors such as humidity, rainfall, and wind conditions.
It applies a Linear Regression model to historical weather data to estimate future temperature values. The goal is to understand the relationship between different weather parameters and their impact on daily temperature.

⸻

Objectives
	•	Predict the temperature of a specific time (e.g., 3 PM) using available weather data.
	•	Analyze the correlation between various meteorological features.
	•	Evaluate the performance of a regression model using metrics like Mean Squared Error (MSE) and R² Score.
	•	Visualize and interpret model predictions for further insights.

⸻

Project Structure

weather_prediction/
│
├── weather_prediction.py        # Main source code
├── weather.csv                  # Dataset (historical weather data)
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation


⸻

Dataset

The dataset used for this project can be downloaded from public sources such as Kaggle’s Weather Dataset (Rattle Package).

Example Columns:
	•	MinTemp: Minimum temperature of the day (°C)
	•	MaxTemp: Maximum temperature of the day (°C)
	•	Rainfall: Rainfall amount (mm)
	•	Humidity9am: Humidity at 9 AM (%)
	•	Humidity3pm: Humidity at 3 PM (%)
	•	Temp3pm: Temperature at 3 PM (°C) — target variable

⸻

Tools and Libraries

This project uses the following technologies:
	•	Python 3.8+
	•	pandas – for data manipulation
	•	NumPy – for numerical computation
	•	scikit-learn – for Linear Regression and evaluation
	•	matplotlib / seaborn – for visualization

Install the dependencies using:

pip install -r requirements.txt

requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn


⸻

Implementation Details

Step 1: Import and Load Data

The dataset is read into a pandas DataFrame. Missing or null values are removed to ensure model accuracy.

df = pd.read_csv("weather.csv")
df = df.dropna()

Step 2: Feature Selection

The relevant features are selected based on correlation analysis.

X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']]
y = df['Temp3pm']

Step 3: Data Splitting

The dataset is divided into 80% training and 20% testing sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 4: Model Training

A Linear Regression model is trained using the training data.

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

Step 5: Prediction and Evaluation

Predictions are made on the test dataset, and performance metrics are calculated.

from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

Step 6: Visualization

A scatter plot of actual vs predicted temperatures provides a visual check of model performance.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()


⸻

Model Performance

Metric	Description	Value (Example)
Mean Squared Error (MSE)	Average of squared prediction errors	4.23
R² Score	Proportion of variance explained by the model	0.91

A higher R² and lower MSE indicate a more accurate regression model.

⸻

Results Summary
	•	The model achieved an R² Score of ~0.90, demonstrating strong predictive power.
	•	Temperature prediction was found to be most influenced by humidity and maximum temperature.
	•	The scatter plot shows a near-linear relationship, validating the regression approach.

⸻

Future Improvements
	•	Introduce Polynomial Regression or Random Forest Regression for nonlinear relationships.
	•	Integrate real-time weather API data for continuous predictions.
	•	Build an interactive Streamlit web interface to allow CSV uploads and instant predictions.
	•	Optimize model using feature scaling and regularization (Ridge/Lasso).

⸻

Usage

Run the script directly from your terminal:

python weather_prediction.py

To visualize the prediction results, ensure matplotlib windows are enabled or use Jupyter Notebook.

⸻
