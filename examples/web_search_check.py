from ddgs import DDGS


def search_web(
    query: str,
    max_results: int = 2,
    topic: str = "general",
    time_range: str | None = None,
    country: str | None = None,
) -> list | str:
    try:
        if topic not in {"general", "news"}:
            raise ValueError(
                "topic must be 'general' or 'news'"
            )

        with DDGS() as ddgs:
            if topic == "news":
                results = ddgs.news(
                    query,
                    region=country or "us-en",
                    timelimit=time_range,
                    max_results=max_results,
                )
            else:
                results = ddgs.text(
                    query,
                    region=country or "us-en",
                    timelimit=time_range,
                    max_results=max_results,
                )

        return [
            {
                "title": result.get("title"),
                "url": result.get("url") or result.get("href"),
                "content": (
                    result.get("body")
                    or result.get("excerpt")
                    or result.get("description")
                ),
            }
            for result in results
        ]

    except Exception as e:
        return f"Error: Search failed - {e}"


if __name__ == "__main__":
    print(
        search_web(
            "US inflation",
            topic="news",
            max_results=2,
        )
    )