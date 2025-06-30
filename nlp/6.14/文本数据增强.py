import requests
import hashlib
import random

def baidu_translate(query,appid="20250614002381778", secret_key="tyQj98sbnSaHD7ANPA3p", from_lang='en', to_lang='zh'):
    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5((appid + query + salt + secret_key).encode()).hexdigest()

    params = {
        'q': query,
        'from': from_lang,
        'to': to_lang,
        'appid': appid,
        'salt': salt,
        'sign': sign
    }

    res = requests.get(url, params=params)
    result = res.json()
    print("翻译结果：", result)


baidu_translate("The price is very cheap", from_lang='en', to_lang='zh')
baidu_translate("这个价格非常便宜",from_lang='zh', to_lang='en')