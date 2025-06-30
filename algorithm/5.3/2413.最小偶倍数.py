import math


class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        return (n%2+1)*n
if __name__ == '__main__':
    print(Solution().smallestEvenMultiple(5))