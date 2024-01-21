# House Price Prediction Model in Jupyter Notebook

This project is an implementation of a simple Linear Regression model to predict house prices using the California Housing dataset. The model is implemented in a Jupyter Notebook.

## Project Description

The model predicts house prices by considering various factors such as the average number of rooms per household. The project demonstrates the use of Python's scikit-learn library to implement the Linear Regression model, and matplotlib to visualize the relationship between the actual prices and predicted prices.

The notebook is organized into separate cells, with each cell containing code that performs a specific task, making it easy to understand and follow. Extensive comments and markdown cells are used to explain the purpose and functionality of each cell.

## Prerequisites

To run the notebook, you need to have Jupyter Notebook installed, which comes with the Anaconda distribution of Python. In case you are not using Anaconda, you can install it using pip:

```bash
pip install notebook
```

The project requires the following Python packages:

- numpy==1.21.2
- pandas==1.3.3
- scikit-learn==0.24.2
- matplotlib==3.4.3

You can install these packages using pip:

```sh
pip install -r requirements.txt
```

To run the notebook, navigate to the directory containing the notebook and start the Jupyter Notebook server:
```
jupyter notebook ./house_price_prediction.ipynp
```

This will open a webpage in your default browser that shows the files in the current directory. Click on the house_price_prediction.ipynb notebook to open it.

## The dataset

The model now uses the California Housing dataset instead of the Boston Housing dataset due to ethical considerations. The California Housing dataset considers features such as the median income, average number of rooms, and location to predict the median house price for California districts.
