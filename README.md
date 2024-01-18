# House Price Prediction Model

This project is an implementation of a simple Linear Regression model to predict house prices using the California Housing dataset.

## Project Description

The model predicts house prices by considering various factors such as the average number of rooms per household. The project demonstrates the use of Python's scikit-learn library to implement the Linear Regression model, and matplotlib to visualize the relationship between the actual prices and predicted prices.

The code is written following clean code principles -- it uses meaningful names for variables and functions, and is organized into logical sections with in-depth comments explaining each section for better readability and understanding.

## Prerequisites

The project requires the following Python packages:

- numpy==1.21.2
- pandas==1.3.3
- scikit-learn==0.24.2
- matplotlib==3.4.3

You can install these packages using pip:

```sh
pip install -r requirements.txt
```

To run the model, navigate to the directory containing the script and use the Python command followed by the script name:
```
python ./house_price_prediction.py
```

This command will execute the script and run the model. The output will be printed in the terminal, and you can observe the actual prices vs predicted prices from the generated scatter plot.

## The dataset

The model now uses the California Housing dataset instead of the Boston Housing dataset due to ethical considerations. The California Housing dataset considers features such as the median income, average number of rooms, and location to predict the median house price for California districts.
