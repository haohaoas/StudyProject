import numpy as np
import pandas as pd

# 读取 boston.txt 文件，跳过前22行注释
with open("boston.txt") as f:
    lines = f.readlines()[22:]

# 每两行组合成一个样本
data = []
for i in range(0, len(lines), 2):
    part1 = list(map(float, lines[i].strip().split()))
    part2 = list(map(float, lines[i+1].strip().split()))
    data.append(part1 + part2)

# 转换为数组
data = np.array(data)

# 设置列名
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# 转为 DataFrame
df = pd.DataFrame(data, columns=columns)

# 保存为 CSV
df.to_csv("boston.csv", index=False)

print("保存成功：boston.csv")
