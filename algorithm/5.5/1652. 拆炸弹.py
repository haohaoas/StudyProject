from typing import List


class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        n = len(code)
        if k == 0:
            return [0] * n

        list_code = code * 2
        res = [0] * n

        if k > 0:
            for i in range(n):
                res[i] = sum(list_code[i + 1: i + k + 1])
        else:
            k = -k
            for i in range(n):
                res[i] = sum(list_code[i + n - k: i + n])

        return res
if __name__ == '__main__':
    s = Solution()
    code = [2,4,9,3]
    k = -2
    print(s.decrypt(code, k))