"""
Main Execution Script for ARIMA-DBO-CNN-LSTM Project.
Usage examples:
  python main.py --city beijing
  python main.py --city guangzhou --dbo_pop 12 --dbo_iter 8
  python main.py --city chengdu --all_models
"""

import os
import argparse
import pandas as pd
import numpy as np

from src.data_loader import load_city_split
from src.arima_module import ARIMAForecaster
from src.hybrid_model import ARIMADBOCNNLSTM
from src.baselines import SVMBaseline, RFBaseline, LSTMBaseline
from src.evaluation import calculate_metrics, plot_comparison_and_scatter, plot_metrics_barchart
from src.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="ARIMA-DBO-CNN-LSTM Air Quality Prediction Framework")
    parser.add_argument("--city", type=str, default="beijing", choices=["beijing", "chengdu", "guangzhou", "wuhan"],
                        help="Target city for AQI prediction")
    parser.add_argument("--all_models", action="store_true", default=True,
                        help="Train and evaluate all comparative models (SVM, RF, ARIMA, LSTM, Proposed)")
    parser.add_argument("--dbo_pop", type=int, default=15, help="DBO population size (default 15)")
    parser.add_argument("--dbo_iter", type=int, default=10, help="DBO max iterations (default 10)")
    parser.add_argument("--device", type=str, default=None, help="PyTorch compute device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()


def run_experiment(city: str, run_all: bool = True, dbo_pop: int = 15, dbo_iter: int = 10, device: str = None):
    set_seed(42)
    dev = device or str(get_device())
    print(f"\n=======================================================")
    print(f"  RUNNING AIR QUALITY PREDICTION PIPELINE: {city.upper()}")
    print(f"  Compute Device: {dev.upper()}")
    print(f"=======================================================\n")

    # 1. Load Train / Test data
    train_data, test_data, dates = load_city_split(city, train_ratio=0.8)
    print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples.")

    # ARIMA preset orders from paper Table 1
    arima_presets = {
        "beijing": (3, 1, 1),
        "guangzhou": (2, 1, 2),
        "chengdu": (3, 1, 1),
        "wuhan": (2, 1, 2),
    }
    preset_order = arima_presets.get(city.lower(), None)

    predictions = {}
    metrics_records = []

    # 2. Run Comparative Baselines (if requested)
    if run_all:
        print("\n--- [Model A] Training Linear SVM ---")
        svm = SVMBaseline(window_size=10).fit(train_data)
        svm_preds = svm.predict(test_data)
        predictions["SVM"] = svm_preds
        metrics_records.append({"Model": "SVM", **calculate_metrics(test_data, svm_preds)})

        print("\n--- [Model B] Training Random Forest ---")
        rf = RFBaseline(window_size=10, n_estimators=100).fit(train_data)
        rf_preds = rf.predict(test_data)
        predictions["RF"] = rf_preds
        metrics_records.append({"Model": "RF", **calculate_metrics(test_data, rf_preds)})

        print("\n--- [Model C] Training Standalone ARIMA ---")
        arima_standalone = ARIMAForecaster(order=preset_order).fit(train_data)
        arima_preds = arima_standalone.forecast_test_rolling(test_data)
        predictions["ARIMA"] = arima_preds
        metrics_records.append({"Model": "ARIMA", **calculate_metrics(test_data, arima_preds)})

        print("\n--- [Model D] Training Standalone LSTM ---")
        lstm = LSTMBaseline(window_size=10, hidden_units=[50, 100, 150], epochs=60, device=dev).fit(train_data)
        lstm_preds = lstm.predict(test_data)
        predictions["LSTM"] = lstm_preds
        metrics_records.append({"Model": "LSTM", **calculate_metrics(test_data, lstm_preds)})

        print("\n--- [Model H] Training ARIMA-CNN-LSTM (without DBO) ---")
        hybrid_nodbo = ARIMADBOCNNLSTM(arima_order=preset_order, use_dbo=False, device=dev)
        hybrid_nodbo.fit(train_data, verbose=False)
        nodbo_preds, _, _ = hybrid_nodbo.predict(test_data)
        predictions["ARIMA-CNN-LSTM"] = nodbo_preds
        metrics_records.append({"Model": "ARIMA-CNN-LSTM", **calculate_metrics(test_data, nodbo_preds)})

    # 3. Run Proposed ARIMA-DBO-CNN-LSTM Model (Model J)
    print("\n--- [Model J] Training Proposed ARIMA-DBO-CNN-LSTM ---")
    proposed = ARIMADBOCNNLSTM(
        arima_order=preset_order,
        use_dbo=True,
        dbo_pop_size=dbo_pop,
        dbo_max_iter=dbo_iter,
        device=dev,
    )
    proposed.fit(train_data, verbose=True)
    proposed_preds, linear_part, residual_part = proposed.predict(test_data)
    predictions["ARIMA-DBO-CNN-LSTM"] = proposed_preds
    metrics_records.append({"Model": "ARIMA-DBO-CNN-LSTM", **calculate_metrics(test_data, proposed_preds)})

    # 4. Summary Results Table
    results_df = pd.DataFrame(metrics_records)
    print(f"\n=======================================================")
    print(f"        EVALUATION METRICS TABLE ({city.upper()})")
    print(f"=======================================================")
    print(results_df.to_string(index=False))

    # Save metrics CSV
    os.makedirs("results", exist_ok=True)
    csv_file = f"results/{city}_metrics.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\nMetrics saved to: {csv_file}")

    # 5. Generate Figures
    comp_plot_path = f"results/figures/{city}_model_comparison.png"
    plot_comparison_and_scatter(
        y_true=test_data,
        predictions=predictions,
        city_name=city,
        save_path=comp_plot_path,
    )

    bar_plot_path = f"results/figures/{city}_metrics_barchart.png"
    plot_metrics_barchart(results_df, city_name=city, save_path=bar_plot_path)

    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        city=args.city,
        run_all=args.all_models,
        dbo_pop=args.dbo_pop,
        dbo_iter=args.dbo_iter,
        device=args.device,
    )
