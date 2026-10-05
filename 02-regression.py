import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preparation
    """)
    return


@app.cell
def _():
    import numpy as np 
    import pandas as pd 

    return (pd,)


@app.cell
def _(pd):
    df=pd.read_csv('data/raw/car_features_and_msrp.csv')
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It seems that features don't follow a certain case, converting all to snake case.
    """)
    return


@app.cell
def _(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    converting all columns with string values to snake_case too.
    """)
    return


@app.cell
def _(df):
    df.dtypes[df.dtypes == 'python'].index
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
