from typing import List
from sortedcontainers import SortedList

class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        res = []
        sl = SortedList()
        for i in range(len(nums)):
            sl.add(nums[i])
            if i >= k - 1:
                beauty = sl[x - 1]
                res.append(beauty if beauty < 0 else 0)
                sl.remove(nums[i - k + 1])
        return res
print(Solution().getSubarrayBeauty(nums = [-3,1,2,-3,0,-3], k = 2, x = 1))
