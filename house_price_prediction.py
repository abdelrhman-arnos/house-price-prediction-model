import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    # Load the California Housing Data from sklearn's datasets.
    housing = datasets.fetch_california_housing()

    # Turn the dataset into a DataFrame for easier manipulation.
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data["MedHouseVal"] = housing.target

    # Explore the dataset and look at the first few rows of the DataFrame to understand the data.
    print(data.head())

    # Identify the feature variable(s) and target variable.
    # In this case, we choose "AveRooms" (average number of rooms per dwelling) as feature and "MedHouseVal" as target.
    X = pd.DataFrame(np.c_[data["AveRooms"]], columns=["AveRooms"])
    Y = data["MedHouseVal"]

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