import os
import matplotlib.pyplot as plt
import shap
import pandas as pd


def save_plot(fig, path: str, formats=("png", "svg")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", format=fmt)


def save_shap_summary(model, X: pd.DataFrame, out_dir: str = "docs/plots"):
    os.makedirs(out_dir, exist_ok=True)
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    save_plot(fig, os.path.join(out_dir, "shap_summary"))
    plt.close(fig)


def save_mermaid_to_png(mmd_file: str, output_file: str):
    """
    Convert Mermaid MMD file to PNG using Mermaid CLI.
    Requires `mmdc` (Mermaid CLI) installed.
    """
    os.system(f"mmdc -i {mmd_file} -o {output_file} --quiet")