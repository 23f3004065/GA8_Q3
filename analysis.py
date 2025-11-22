# 23f3004065@ds.study.iitm.ac.in
# MARIMO NOTEBOOK (analysis.py)

import marimo

app = marimo.App()


# ---------------------------------------------------------
# Cell 1 — load data
# ---------------------------------------------------------
@app.cell
def cell1():
    """
    Data flow:
    - This cell loads the Iris dataset into `df`
    - Next cells depend on df
    """
    import pandas as pd
    from sklearn import datasets

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris


# ---------------------------------------------------------
# Cell 2 — widgets + variable dependency on df
# ---------------------------------------------------------
@app.cell
def cell2(df, iris):
    """
    Data flow:
    - Depends on df (from cell 1)
    - Creates interactive slider `target_slider`
    """
    import marimo as mo
    target_slider = mo.ui.slider(0, int(df["target"].max()), 1, label="Target class:")
    return target_slider


# ---------------------------------------------------------
# Cell 3 — dynamic markdown, updates when slider changes
# ---------------------------------------------------------
@app.cell
def cell3(df, iris, target_slider):
    """
    Data flow:
    - Depends on df and target_slider
    - Produces dynamic markdown
    """
    import marimo as mo
    t = target_slider.value
    subset = df[df["target"] == t]

    md = f"""
    ### Dynamic Summary for Class {t} — *{iris.target_names[t]}*
    - Sample count: **{len(subset)}**
    - Mean sepal length: **{subset['sepal length (cm)'].mean():.3f}**
    - Mean sepal width: **{subset['sepal width (cm)'].mean():.3f}**
    """
    return mo.md(md)


# ---------------------------------------------------------
# Cell 4 — plotting and more dependency
# ---------------------------------------------------------
@app.cell
def cell4(df, target_slider):
    """
    Data flow:
    - Depends on df and slider
    - Updates scatter plot interactively
    """
    import marimo as mo
    import matplotlib.pyplot as plt

    t = target_slider.value
    subset = df[df["target"] == t]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(subset["sepal length (cm)"], subset["sepal width (cm)"], color="C1")
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("sepal width (cm)")
    ax.set_title(f"Scatter plot for class {t}")

    return mo.pyplot(fig)


# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run()
