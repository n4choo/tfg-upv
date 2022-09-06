import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd


def res_total_visits(df_res: pd.DataFrame) -> pd.DataFrame:
    df_res = df_res.dropna(
        subset=["date_range_start", "date_range_end", "sg_mp__visits_by_day"]
    )
    df_res["date_range_start"] = pd.to_datetime(df_res["date_range_start"])
    df_res["date_range_end"] = pd.to_datetime(df_res["date_range_end"])
    df_res["sg_mp__visits_by_day"] = df_res["sg_mp__visits_by_day"].map(
        json.loads
    )
    df_res_sort = df_res.sort_values(by=["date_range_start"])
    df_res_sort["Date"] = df_res_sort.apply(_create_days_col, axis=1)
    df_res_exploded = (
        df_res_sort.explode(["Date", "sg_mp__visits_by_day"])
        .reset_index(drop=True)
        .groupby(by=["Date"], as_index=False)["sg_mp__visits_by_day"]
        .sum()
    )
    return df_res_exploded


def _filtered_visits_country_df(safe_path: Path):
    df_safe = pd.read_csv(safe_path)
    df_safe = df_safe.dropna(
        subset=["date_range_start", "date_range_end", "sg_mp__visits_by_day"]
    )
    df_safe["sg_c__category_tags"].fillna("", inplace=True)
    df_filtered = df_safe.loc[
        df_safe["sg_c__category_tags"].str.contains("Food")
    ]
    return df_filtered


def _filtered_safe_df(
    in_safe_df: Path,
    category_tag: str,
    output_path: Path,
):
    df_safe = pd.read_csv(in_safe_df)
    df_safe = df_safe.dropna(
        subset=["date_range_start", "date_range_end", "sg_mp__visits_by_day"]
    )
    df_filtered = df_safe[df_safe["sg_c__category_tags"] == category_tag]
    df_filtered.to_csv(output_path)


def _create_days_col(row):
    return (
        pd.date_range(
            start=row["date_range_start"],
            end=row["date_range_end"] - timedelta(days=1),
        )
        .to_pydatetime()
        .tolist()
    )


def _generate_country_res_df(
    in_df_path: Path,
    out_df_path: Path,
    country: str,
    df_total_visits: pd.DataFrame,
):
    df_res = pd.read_csv(in_df_path)
    df_res = df_res.dropna(
        subset=["date_range_start", "date_range_end", "sg_mp__visits_by_day"]
    )
    df_res["date_range_start"] = pd.to_datetime(df_res["date_range_start"])
    df_res["date_range_end"] = pd.to_datetime(df_res["date_range_end"])
    df_res["sg_mp__visits_by_day"] = df_res["sg_mp__visits_by_day"].map(
        json.loads
    )
    df_res_sort = df_res.sort_values(by=["date_range_start"])
    df_res_sort["Date"] = df_res_sort.apply(_create_days_col, axis=1)
    df_res_exploded = (
        df_res_sort.explode(["Date", "sg_mp__visits_by_day"])
        .reset_index(drop=True)
        .groupby(by=["Date"], as_index=False)["sg_mp__visits_by_day"]
        .sum()
    )
    df_res_exploded["country"] = country
    df_res_exploded["normalized_visits"] = (
        df_res_exploded["sg_mp__visits_by_day"]
        * 100
        / df_total_visits["sg_mp__visits_by_day"]
    )
    df_res_exploded.to_csv(out_df_path, index=False)


if __name__ == "__main__":
    restaurant_category_type = "Russian Food"
    country = "Russia"
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_csv:
        _filtered_safe_df(
            Path("data/safegraph_ny/NY.csv"),
            restaurant_category_type,
            Path(temp_csv.name),
        )
        filtered_total_visits = _filtered_visits_country_df(
            Path("data/safegraph_ny/NY.csv")
        )
        df_total_visits = res_total_visits(filtered_total_visits)
        _generate_country_res_df(
            Path(temp_csv.name),
            Path(f"data/streamlit_data/visits/NY-{country}-to-days.csv"),
            country,
            df_total_visits,
        )
