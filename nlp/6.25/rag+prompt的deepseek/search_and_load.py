import requests
from bs4 import BeautifulSoup
from url_search import search_urls

def is_page_valid(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f" 网页请求失败：{url} 状态码：{resp.status_code}")
            return False

        soup = BeautifulSoup(resp.content, "html.parser")
        text = soup.get_text(separator='\n').strip()

        if len(text) < 500:
            print(f"网页内容太少：{url}")
            return False

        print(f" 网页有效：{url}")
        return True

    except Exception as e:
        print(f" 网页请求异常：{url}，原因：{e}")
        return False


def load_web_docs(url_list):
    docs = []
    for url in url_list:
        if is_page_valid(url):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.content, "html.parser")
                text = soup.get_text(separator='\n').strip()

                docs.append(text)
                print(f" 成功抓取：{url}")

            except Exception as e:
                print(f" 抓取失败：{url}，原因：{e}")
        else:
            print(f" 跳过无效页面：{url}")

    return docs


def auto_build_docs(query):
    url_list = search_urls(query)
    print(f"🔍 搜索到的URL列表：{url_list}")

    if not url_list:
        print(" 没有有效链接，尝试更换关键词。")
        return []

    docs = load_web_docs(url_list)
    print(f" 成功加载 {len(docs)} 个文档。")

    return docs