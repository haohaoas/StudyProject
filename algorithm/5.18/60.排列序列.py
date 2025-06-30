import heapq
from typing import List



class Solution:
  def getPermutation(self, n: int, k: int) -> str:
    nums = list(range(1, n + 1))
    factorial = [1] * (n + 1)
    for i in range(1, n + 1):
        factorial[i] = factorial[i - 1] * i
    result = []
    k -= 1
    while n > 0:
        index = k // factorial[n - 1]
        result.append(str(nums[index]))
        k %= factorial[n - 1]
        nums.pop(index)
        n -= 1
    return ''.join(result)

if __name__ == '__main__':
    s = Solution()
    print(s.getPermutation(4,9))