# 23f3004065@ds.study.iitm.ac.in
# analysis.py
# Marimo-style interactive notebook script (Jupyter / VSCode interactive)
#
# This file demonstrates:
# - email as comment (above)
# - multiple cells with variable dependencies
# - an ipywidgets slider controlling analysis
# - dynamic Markdown output based on widget state
# - comments documenting data flow between cells

# %% Cell 1 — imports and data load (base dataset)
# Data flow: this cell loads the data into `df`. Later cells depend on `df`.
import pandas as pd
import numpy as np
from sklearn import datasets
from IPython.display import display, Markdown, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets

# Load Iris dataset into a DataFrame for easy exploration
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map(lambda i: iris.target_names[i])

# Basic sanity print (will show when you run the cell)
print("Data loaded: df.shape =", df.shape)
df.head()

# %% Cell 2 — feature selection & derived variables
# Data flow: this cell defines `x_col` and `y_col` (dependent variables),
# and computes an initial statistic `base_corr` used by UI.
# Later cells read x_col/y_col/base_corr.
x_col = iris.feature_names[0]   # sepal length (cm) by default
y_col = iris.feature_names[1]   # sepal width (cm) by default

# Derived variable: base correlation (whole dataset)
base_corr = df[[x_col, y_col]].corr().iloc[0, 1]

# Documenting data flow:
# - df (from cell 1) -> this cell uses df to compute base_corr
# - UI/slider (cell 3) will use x_col/y_col and df to recompute filtered stats
print(f"Default features: x_col={x_col}, y_col={y_col}, base_corr={base_corr:.4f}")

# %% Cell 3 — interactive slider widget & dynamic markdown + plot
# Data flow:
# - This cell reads `df`, `x_col`, `y_col` (from previous cells)
# - The slider controls `min_target` which filters rows by `target` label
# - On change, we update a dynamic Markdown summary and a scatter plot

# Slider to choose which class (target) to focus on -- represent as integer 0..2
min_target_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=int(df['target'].max()),
    step=1,
    description='Target class:',
    continuous_update=False
)

# Dropdowns to choose features interactively (additional interactivity)
x_dropdown = widgets.Dropdown(options=iris.feature_names, value=x_col, description='X feature:')
y_dropdown = widgets.Dropdown(options=iris.feature_names, value=y_col, description='Y feature:')

# Output widget areas for dynamic markdown and plot
md_out = widgets.Output()
plot_out = widgets.Output()

def update_ui(change=None):
    """
    Recompute filters, statistics, and update plot + markdown.
    This function reads df, uses the current widget states (target, x, y)
    and then writes to md_out and plot_out.
    """
    # read current widget state
    tgt = min_target_slider.value
    x_feat = x_dropdown.value
    y_feat = y_dropdown.value

    # Data filtering: select rows where target == tgt (this is the variable dependency)
    filtered = df[df['target'] == tgt]

    # Compute statistics on filtered data
    mean_x = filtered[x_feat].mean()
    mean_y = filtered[y_feat].mean()
    corr = filtered[[x_feat, y_feat]].corr().iloc[0, 1]

    # Update dynamic markdown
    with md_out:
        clear_output(wait=True)
        display(Markdown(f"### Dynamic summary (class = **{tgt}** — *{iris.target_names[tgt]}*)"))
        display(Markdown(f"- Samples in selected class: **{len(filtered)}**"))
        display(Markdown(f"- Mean **{x_feat}** = **{mean_x:.3f}**"))
        display(Markdown(f"- Mean **{y_feat}** = **{mean_y:.3f}**"))
        display(Markdown(f"- Correlation ({x_feat}, {y_feat}) in selected class = **{corr:.3f}**"))
        display(Markdown(f"_Base correlation across full dataset for ({x_feat}, {y_feat})_ = **{df[[x_feat, y_feat]].corr().iloc[0,1]:.3f}**"))

    # Update the scatter plot
    with plot_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6,4))
        # Plot all points faintly for context
        ax.scatter(df[x_feat], df[y_feat], color='lightgray', alpha=0.5, label='all classes')
        # Highlight filtered points
        ax.scatter(filtered[x_feat], filtered[y_feat], color='C1', edgecolor='k', s=60, label=f'class {tgt}')
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f"{x_feat} vs {y_feat} (class {tgt}: {iris.target_names[tgt]})")
        ax.legend()
        plt.show()

# Wire up widget observers
min_target_slider.observe(update_ui, names='value')
x_dropdown.observe(update_ui, names='value')
y_dropdown.observe(update_ui, names='value')

# Initial display (renders the widgets and outputs in the notebook)
display(widgets.HBox([min_target_slider, x_dropdown, y_dropdown]))
display(md_out)
display(plot_out)

# Call once to populate initial outputs
update_ui()

# %% Cell 4 — notes & documented data flow summary (final cell)
# Data flow summary (documenting the pipeline):
# - Cell 1: loads raw dataset into `df`
# - Cell 2: sets default features `x_col`, `y_col` and computes `base_corr`
# - Cell 3: uses widgets (slider + dropdowns) to filter `df` into `filtered`,
#           computes summary stats and correlation on `filtered`, then updates
#           dynamic Markdown and a Matplotlib scatter plot.
#
# This cell is intentionally a non-executing documentation cell that the reader
# can use to understand how variables propagate through the notebook.
#
# END OF NOTEBOOK
