import shap
import matplotlib.pyplot as plt
from parser import get_args
args = get_args()

def plot_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f'{args.output_path}/shap_summary_plot.png')