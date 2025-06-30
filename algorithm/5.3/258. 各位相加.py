import math


class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
            x=math.log(n,2)
            if x==int(x):
                return True
            else:
                return False

if __name__ == '__main__':
    s = Solution()
    print(s.isPowerOfTwo(536870912))