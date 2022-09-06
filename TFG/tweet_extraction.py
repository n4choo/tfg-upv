import datetime
from typing import Any, Optional

import pandas as pd
from get_location import get_location_id
from pysentimiento.preprocessing import preprocess_tweet
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened

from config import ACADEMIC_BEARER_TOKEN

LOCATION = "new york"
# Start and end times must be in UTC
START_DATE = datetime.datetime(2020, 6, 30, 0, 0, 0, 0, datetime.timezone.utc)
END_DATE = datetime.datetime(2021, 12, 30, 0, 0, 0, 0, datetime.timezone.utc)


def _get_tuple_date(
    start_date: datetime.date, end_date: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    date_range = (
        pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    )
    return [
        (date_range[i], date_range[i + 1]) for i in range(len(date_range) - 1)
    ]


def _get_model():
    # model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
    )


def _transform_roberta_results(t_results):
    t_results = t_results[0]
    label = t_results["label"]
    if label == "Neutral":
        return 0
    else:
        if label == "Positive":
            if t_results["score"] < 0.70:
                return 1
            else:
                return 2
        else:
            if t_results["score"] < 0.70:
                return -1
            else:
                return -2


def _get_pos_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    pos_df = df[df["sentiment_value"] > 0]
    pos_df = pos_df.groupby(["date"], as_index=False).sum()
    return pos_df["sentiment_value"]


def _get_neg_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    neg_df = df[df["sentiment_value"] < 0]
    neg_df = neg_df.groupby(["date"], as_index=False).sum()
    return neg_df["sentiment_value"]


def _get_sentiment_daily(df: pd.DataFrame, country: str):
    pos_sentiment = _get_pos_sentiment(df)
    neg_sentiment = _get_neg_sentiment(df)
    df = df.groupby(["date"], as_index=False).sum()
    df["date"] = pd.to_datetime(df["date"])
    df["country"] = country
    df = df.sort_values(by="date", ignore_index=True)
    df["sentiment_value_norm"] = df["sentiment_value"] / 500
    df["neg_sentiment"] = neg_sentiment
    df["pos_sentiment"] = pos_sentiment
    df.to_csv(
        f"data/streamlit_data/sentiment/ny-{country}-daily-sentiment.csv",
        index=False,
    )


def _transform_twitter_res(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df = df.copy()
    generator = _get_model()
    df["text_prepro"] = df["text"].apply(preprocess_tweet, **{"lang": "en"})
    df["sentiment_trans"] = df["text_prepro"].map(generator)
    df["sentiment_value"] = df["sentiment_trans"].map(
        _transform_roberta_results
    )
    df.to_csv(
        f"data/twitter/ny-{country}-tweets.csv",
        index=False,
    )

    return df


def yield_temporal_interval(
    initial_date: datetime.date, end_date: datetime.date
):
    diff = (end_date - initial_date).days
    for _ in range(diff):
        if initial_date == end_date:
            break
        next_date = initial_date + datetime.timedelta(days=1)
        yield initial_date, next_date
        initial_date = next_date + datetime.timedelta(days=1)


def _get_twitter_api() -> Twarc2:
    # Your bearer token here
    return Twarc2(bearer_token=ACADEMIC_BEARER_TOKEN)


# search_results is a generator, max_results is max tweets per page, 100 max for full archive search with all expansions.
def _call_api_tweets(
    twitter_api: Twarc2,
    location: str,
    query: str,
    start_time: datetime.date,
    end_time: datetime.date,
    max_results: int = 10,
    n_pages: Optional[int] = None,
):
    search_results = twitter_api.search_all(
        query=query,
        start_time=start_time,
        end_time=end_time,
        max_results=max_results,
    )
    result_dict: dict[str, Any] = {
        "text": [],
        "date": [],
        "location": [],
        "username": [],
    }

    # Get all results page by page:
    for i, page in enumerate(search_results):
        # Do something with the whole page of results:
        # print(page)
        # or alternatively, "flatten" results returning 1 tweet at a time, with expansions inline:
        if n_pages is not None and i == (n_pages):
            break
        for tweet in ensure_flattened(page):
            # Do something with the tweet
            result_dict["text"].append(tweet["text"])
            result_dict["date"].append(tweet["created_at"].split("T")[0])
            result_dict["location"].append(location)
            result_dict["username"].append(tweet["author"]["username"])
    result_df = pd.DataFrame(result_dict)
    return pd.DataFrame(result_df)


def _get_tweets_df(query: str, country: str) -> pd.DataFrame:
    df_tweets = pd.concat(
        [
            _call_api_tweets(
                _get_twitter_api(),
                LOCATION,
                query,
                ini_date,
                end_date,
                max_results=100,
                n_pages=5,
            )
            for ini_date, end_date in _get_tuple_date(START_DATE, END_DATE)
        ]
    ).reset_index(drop=True)
    df_tweets.to_csv(
        f"data/twitter/ny-{country}-tweets.csv",
        index=False,
    )
    df_tweets["tweet_count"] = 1
    return df_tweets


def main():
    city_id = get_location_id(LOCATION)
    country = "Italy"
    query = f"italy OR Italy or ITALY OR italian OR Italian or ITALIAN lang:en place:{city_id}"
    df_tweets = _get_tweets_df(query, country)
    df_tweets = _transform_twitter_res(df_tweets, country)
    _get_sentiment_daily(df_tweets, country)


if __name__ == "__main__":
    main()
