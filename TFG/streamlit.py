from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import grangercausalitytests


@st.cache
def _read_merge_df(
    path_folder_sent: Path, path_folder_visits: Path
) -> pd.DataFrame:
    df_sent = pd.concat(
        [pd.read_csv(file) for file in path_folder_sent.rglob("*.csv")]
    )
    df_visits = pd.concat(
        [pd.read_csv(file) for file in path_folder_visits.rglob("*.csv")]
    )
    df_visits.rename(
        columns={"Date": "date", "sg_mp__visits_by_day": "visits"},
        inplace=True,
    )
    df_merged = pd.merge(df_sent, df_visits, on=["date", "country"])
    df_merged.drop(
        labels=["tweet_count", "normalized_pos", "normalized_neg"],
        axis=1,
        inplace=True,
    )
    print(df_merged)
    return df_merged


def _plot_trends(
    df_merged: pd.DataFrame, sent_col: str, visits_type: str
) -> go.Figure:
    # sent_ser = df_merged[sent_col].rolling(window=30).mean()
    # visits_ser = df_merged[visits_type].rolling(window=7).mean()

    df = pd.DataFrame().assign(
        **{
            sent_col: df_merged[sent_col],
            visits_type: df_merged[visits_type],
        }
    )
    df["sent_pct"] = df[sent_col].pct_change()
    df["visits_pct"] = df[visits_type].pct_change()
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    sent_ser = pd.Series(
        np.convolve(
            df[sent_col],
            np.ones(3) / 3,
            mode="valid",
        )
    )
    sent_ser.index = df.index[2:]
    visits_series = pd.Series(
        np.convolve(
            df[visits_type],
            np.ones(3) / 3,
            mode="valid",
        )
    )
    visits_series.index = df.index[2:]
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[visits_type],
            name="Visits data",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[sent_col],
            name="Sentiment data",
        ),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="Double Y Axis Example")

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Visits</b> yaxis", secondary_y=False)
    fig.update_yaxes(
        title_text="<b>Sentiment</b> yaxis title", secondary_y=True
    )

    pct_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pct_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sent_pct"],
            name="Sentiment data",
        ),
        secondary_y=True,
    )
    pct_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["visits_pct"],
            name="Visits data",
        ),
        secondary_y=True,
    )
    # Create second plot with pyplot
    plt_fig, ax = plt.subplots()
    ax.scatter(df["sent_pct"], df["visits_pct"])
    ax.set_xlabel("Change in visits")
    ax.set_ylabel("Change in sentiments")
    pct_corr = df["visits_pct"].corr(df["sent_pct"])
    corr = df[visits_type].corr(df[sent_col])
    return fig, plt_fig, pct_fig, pct_corr, corr


def _filter_df(
    df: pd.DataFrame,
    country: str,
    time_freq: str,
    sent_type: str,
    visits_type: str,
) -> pd.DataFrame:
    time_freq_transform: dict[str, str] = {
        "daily": "D",
        "weekly": "W",
        "half-monthly": "SMS",
        "monthly": "M",
    }
    df = df[df["country"] == country]
    df["date"] = pd.to_datetime(df["date"].copy())
    df[visits_type] = df[visits_type].rolling(window=14).mean()
    df[sent_type] = df[sent_type].rolling(window=14).mean()
    df.dropna(inplace=True)
    df = df.groupby(
        pd.Grouper(key="date", axis=0, freq=time_freq_transform[time_freq])
    ).sum()
    if time_freq == "weekly" or time_freq == "half-monthly":
        df.drop(df.head(1).index, inplace=True)
        df.drop(df.tail(1).index, inplace=True)
    if time_freq == "monthly":
        df.drop(df.head(1).index, inplace=True)
    return df


def pd_crosscorr(datax, datay, lag=0):
    """Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def get_cross_corr(
    df: pd.DataFrame, sent_type: str, visits_type: str, time_frame: str
):
    if time_frame == "daily":
        maxlags = 30
    elif time_frame == "weekly":
        maxlags = 5
    elif time_frame == "half-monthly":
        maxlags = 2
    elif time_frame == "monthly":
        maxlags = 1

    sent = df[sent_type]
    visits = df[visits_type]
    cros_corr = [
        pd_crosscorr(visits, sent, lag=lag) for lag in range(maxlags + 1)
    ]
    fig = px.bar(x=list(range(maxlags + 1)), y=cros_corr)

    return fig


if __name__ == "__main__":
    df_visits_sent = _read_merge_df(
        Path("data/streamlit_data/sentiment"),
        Path("data/streamlit_data/visits"),
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        country_frame = st.selectbox(
            "Select country",
            (
                "Mexico",
                "Italy",
                "Russia",
                "China",
            ),
        )
    with col2:
        time_frame = st.selectbox(
            "Select time frequency",
            ("daily", "weekly", "half-monthly", "monthly"),
        )
    with col3:
        sent_type_mapping = {
            "Normalized mean": "sentiment_value_norm",
            "Mean": "sentiment_value",
            "Positive": "pos_sentiment",
            "Negative": "neg_sentiment",
        }
        sent_type = st.selectbox(
            "Select sentiment",
            (
                "Normalized mean",
                "Mean",
                "Positive",
                "Negative",
                "Norm Pos",
                "Norm Neg",
            ),
        )
        sent_type = sent_type_mapping[sent_type]
    with col4:
        visits_type_mapping = {
            "Regular Visits": "visits",
            "Normalized Visits": "normalized_visits",
        }
        visits_type = st.selectbox(
            "Select visits",
            ("Regular Visits", "Normalized Visits"),
        )
        visits_type = visits_type_mapping[visits_type]

    df_visits_sent = _filter_df(
        df_visits_sent, country_frame, time_frame, sent_type, visits_type
    )
    fig, plt_fig, pct_fig, pct_corr, corr = _plot_trends(
        df_visits_sent, sent_type, visits_type
    )
    fig_shift_corr = get_cross_corr(
        df_visits_sent, sent_type, visits_type, time_frame
    )
    st.plotly_chart(fig)
    st.plotly_chart(fig_shift_corr)
    granger_dict = grangercausalitytests(
        df_visits_sent[[sent_type, visits_type]].pct_change().dropna(),
        maxlag=[2],
    )
    print(granger_dict[2][1][0].summary2())
