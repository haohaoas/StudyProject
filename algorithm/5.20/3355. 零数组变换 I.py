from typing import List


class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        for query in queries:
            L, R = query
            for i in range(L, R + 1):
                if nums[i] > 0:
                    nums[i] -= 1
        for num in nums:
            if num != 0:
                return False
        return True

if __name__ == '__main__':
    print(Solution().isZeroArray(nums = [3], queries = [[0,0],[0,0]]))