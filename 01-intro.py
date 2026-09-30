import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1 - Introduction to Machine Learning
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q1. Pandas version
    """)
    return


@app.cell
def _(pd):
    pd.__version__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q2. Records count
    How many records are in the dataset?
    """)
    return


@app.cell
def _(pd):
    raw_data_path = "data/raw/car_fuel_efficiency_2026.csv"
    df = pd.read_csv(raw_data_path)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q3. Fuel types
    How many fuel types are presented in the dataset?
    """)
    return


@app.cell
def _(df):
    df["fuel_type"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q4. Missing values
    How many columns in the dataset have missing values?
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q5. Max fuel efficiency
    What's the maximum fuel efficiency of cars from Asia?
    """)
    return


@app.cell
def _(df):
    df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q6. Median value of horsepower

    1. Find the median value of the horsepower column in the dataset.
    2. Next, calculate the most frequent value of the same horsepower column.
    3. Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
    4. Now, calculate the median value of horsepower once again.

    Has it changed?
    """)
    return


@app.cell
def _(df):
    df["horsepower"].describe()
    return


@app.cell
def _(df):
    df["horsepower"].median()
    return


@app.cell
def _(df):
    df["horsepower"].mode()[0]
    return


@app.cell
def _(df):
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mode()[0])
    return


@app.cell
def _(df):
    df['horsepower'].median()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q7. Sum of weights
    1. Select all the cars from Asia
    2. Select only columns vehicle_weight and model_year
    3. Select the first 7 values
    4. Get the underlying NumPy array. Let's call it X.
    5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
    6. Invert XTX.
    7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
    8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
    9. What's the sum of all the elements of the result?
    """)
    return


@app.cell
def _(df):
    df["origin"]
    return


@app.cell
def _(df):
    asian_cars = df[df["origin"] == "Asia"]
    return (asian_cars,)


@app.cell
def _(asian_cars):
    asian_cars.head(7)
    return


@app.cell
def _(asian_cars):
    X = asian_cars[["vehicle_weight", "model_year"]].head(7)
    return (X,)


@app.cell
def _(np):
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    return (y,)


@app.cell
def _(X):
    XTX = X.T @ X
    return (XTX,)


@app.cell
def _(XTX, np):
    XTX_inverse = np.linalg.inv(XTX)
    return (XTX_inverse,)


@app.cell
def _(X, XTX_inverse):
    XTX_inverseXT = (XTX_inverse @ X.T)
    return (XTX_inverseXT,)


@app.cell
def _(XTX_inverseXT, y):
    w = XTX_inverseXT * y
    print(w)
    return (w,)


@app.cell
def _(w):
    w.sum().sum()
    return


if __name__ == "__main__":
    app.run()
