str1='AI⼈⼯智能进阶班'
# 字符串转二进制
binary_str = bin(int.from_bytes(str1.encode(), 'big'))[2:]
print(f'字符串转二进制 {binary_str}')

string1 = "AI⼈⼯智能进阶班"
bytes1 = string1.encode()
print(f"字符串 '{string1}' 转换为二进制 bytes 类型的结果: {bytes1}")

bytes2 = b"AI python"
string2 = bytes2.decode()
print(f"二进制 bytes 数据 {bytes2} 转换为字符串 str 类型的结果: {string2}")