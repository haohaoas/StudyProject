from functools import cmp_to_key
from typing import List


class Solution:
    def crackPassword(self, password: List[int]) -> str:
        strs = list(map(str, password))

        # 自定义比较函数
        def compare(a, b):
            if a + b < b + a:
                return -1
            elif a + b > b + a:
                return 1
            else:
                return 0

        # 排序
        strs.sort(key=cmp_to_key(compare))
        result = ''.join(strs)
        return result if result else '0'


if __name__ == '__main__':
    print(Solution().crackPassword([0, 3, 30, 34, 5, 9]))
