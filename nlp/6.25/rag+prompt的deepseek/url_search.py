from serpapi import GoogleSearch

def search_urls(query, num_results=5):
    params = {
      "engine": "baidu",   # 百度搜索
      "q": query,
      "api_key": "6ec904be856726cc61956d62cd44d0d88f0a89e3cedc7c47cbcb98d245240239",
      "num": num_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    urls = []

    if "organic_results" in results:
        for result in results["organic_results"]:
            urls.append(result.get("link"))

    return urls
