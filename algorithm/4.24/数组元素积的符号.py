from typing import List


class Solution:
    def arraySign(self, nums: List[int]) -> int:
        res = 1
        for i in nums:
            if i == 0:
                return 0
            elif i < 0:
                res *= -1
        return res
if __name__ == '__main__':
    nums = [1,5,0,2,-3]
    print(Solution().arraySign(nums))
