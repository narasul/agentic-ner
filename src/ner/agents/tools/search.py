import os
from typing import List

from langchain_community.document_loaders import BraveSearchLoader
from tavily import TavilyClient


def search_with_brave(queries: List[str]) -> str:
    api_key = os.environ.get("BRAVE_API_KEY") or ""

    answer = ""
    for query in queries:
        try:
            loader = BraveSearchLoader(
                query=query, api_key=api_key, search_kwargs={"count": 5}
            )
            docs = loader.load()
        except Exception as err:
            print(f"Error while calling search API: {str(err)}")
            return (
                "Sorry, there is an issue while using the search tool with given input"
            )

        context = ""
        for doc in docs:
            context += f"--\n{doc.page_content}\n"

        answer += f"Search results for '{query}':\n{context}\n-----\n"

    with open("results.txt", "w") as file:
        file.write(answer)

    return answer


def search_with_tavily(queries: List[str]) -> str:
    api_key = os.environ.get("TAVILY_API_KEY") or ""
    tavily_client = TavilyClient(api_key=api_key)

    answer = ""
    for query in queries:
        try:
            response = tavily_client.search(query=query, include_answer=True)
        except Exception as err:
            print(f"Error while calling search API: {str(err)}")
            return (
                "Sorry, there is an issue while using the search tool with given input"
            )

        answer += (
            f"Search results for '{query}':\n{response.get('answer', '')}\n-----\n"
        )

    return answer


if __name__ == "__main__":
    print(
        search_with_tavily(
            queries=["What is BRCA1?", "What kind of car is Porsche 911 GT?"]
        )
    )
