from functools import reduce
from operator import xor


class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        nums=[start+2*i for i in range(n)]
        return reduce(xor,nums)
if __name__ == '__main__':
    s=Solution()
    print(s.xorOperation(n = 5, start = 0))