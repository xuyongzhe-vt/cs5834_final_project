import seaborn as sns
from urban_project.data_loader import load_and_process_data
from urban_project.analyser.analyzer import analyse_and_draw_plots
from urban_project.train_model.processor import train_model_and_compare


SNS_STYLE = "whitegrid"

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_and_process_data("input")

    # Preview the data
    if df is not None:
        sns.set_theme(style=SNS_STYLE)

        # 1. Run Analyses
        analyse_and_draw_plots(df)

        # 2. train model
        train_model_and_compare(df)
    else:
        print("Data load failed.")
