import requests
import pandas as pd

url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.300251&fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&end=20500101&lmt=1000"
resp = requests.get(url)
data = resp.json()
if data.get("data") and data["data"].get("klines"):
    klines = data["data"]["klines"]
    df = pd.DataFrame([k.split(',') for k in klines], columns=["date","open","close","high","low","volume","turnover","amplitude","chg","percent","turnover_rate"])
    print(df.tail(5))
else:
    print("未获取到有效数据，请检查接口参数或更换数据源")