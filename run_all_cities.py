"""
Batch Runner for all four cities: Beijing, Chengdu, Guangzhou, Wuhan.
Generates comprehensive cross-city comparison tables and summary charts.
"""

import os
import pandas as pd
from main import run_experiment


def run_all_cities(dbo_pop: int = 12, dbo_iter: int = 8):
    cities = ["beijing", "chengdu", "guangzhou", "wuhan"]
    all_city_results = []

    for city in cities:
        print(f"\n>>>>>>>>>>>> PROCESSING CITY: {city.upper()} <<<<<<<<<<<<")
        df_res = run_experiment(
            city=city,
            run_all=True,
            dbo_pop=dbo_pop,
            dbo_iter=dbo_iter,
        )
        df_res["City"] = city.capitalize()
        all_city_results.append(df_res)

    master_df = pd.concat(all_city_results, ignore_index=True)
    master_path = "results/master_comparison_all_cities.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\n=======================================================")
    print(f"  MASTER BENCHMARK REPORT COMPLETED ACROSS ALL 4 CITIES")
    print(f"  Saved to: {master_path}")
    print(f"=======================================================\n")
    print(master_df.to_string(index=False))


if __name__ == "__main__":
    run_all_cities()
