import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    # Load the Boston Housing Data from sklearn's datasets.
    boston = datasets.load_boston()

    # Turn the dataset into a DataFrame for easier manipulation.
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data["MEDV"] = boston.target

    # Explore the dataset and look at the first few rows of the DataFrame to understand the data.
    print(data.head())

    # Identify the feature variable(s) and target variable.
    # In this case, we might choose "RM" (average number of rooms per dwelling) as feature and "MEDV" as target.
    X = pd.DataFrame(np.c_[data["RM"]], columns=["RM"])
    Y = data["MEDV"]

    # Split our data into training and testing subsets.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=9
    )

    # Define, fit, and run our linear regression model using train data.
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Evaluate the model using the test data.
    Y_pred = model.predict(X_test)

    # Create metrics (like Mean Squared Error) to assess your model's performance.
    mse = metrics.mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error: ", mse)

    # (Optional) We can use Matplotlib to create scatter plot of actual prices vs predicted prices.
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted prices")
    plt.title("Actual prices vs Predicted prices")


# This is the code that runs if your file is called from the terminal.
if __name__ == "__main__":
    main()
