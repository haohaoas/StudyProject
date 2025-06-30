from collections import defaultdict


class Solution:
        def minFlips(self, s: str) -> int:
            n = len(s)
            s = s + s
            res = n
            diff1 = diff2 = 0

            for i in range(2 * n):
                if s[i] != ('0' if i % 2 == 0 else '1'):
                    diff1 += 1
                if s[i] != ('1' if i % 2 == 0 else '0'):
                    diff2 += 1

                if i >= n:
                    if s[i - n] != ('0' if (i - n) % 2 == 0 else '1'):
                        diff1 -= 1
                    if s[i - n] != ('1' if (i - n) % 2 == 0 else '0'):
                        diff2 -= 1

                if i >= n - 1:
                    res = min(res, diff1, diff2)

            return res


if __name__ == '__main__':
    print(Solution().minFlips(s = "01001001101"))