import requests

proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}

url = 'https://file.hankcs.com/hanlp/ner/ner_bert_base_msra_20211227_114712.zip'

response = requests.get(url, proxies=proxies, stream=True)
print("状态码:", response.status_code)