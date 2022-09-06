from twarc.client2 import Twarc2

from config import ACADEMIC_BEARER_TOKEN

t = Twarc2(bearer_token=ACADEMIC_BEARER_TOKEN)


def get_location_id(
    city: str, granularity: str = "city", verbose: bool = False
) -> str:
    """Get location ID from a city name"""
    geo = t.geo(query=city, granularity=granularity, max_results=3)
    if verbose:
        for result in geo["result"]["places"]:
            print(result)
            print("*" * 100)
    return geo["result"]["places"][0]["id"]


if __name__ == "__main__":
    get_location_id("new york", verbose=True)
