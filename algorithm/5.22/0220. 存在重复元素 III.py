from typing import List
from bisect import bisect_left, bisect_right

from sortedcontainers import SortedList


class Solution:

    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        if valueDiff < 0 or indexDiff < 1 or len(nums) < 2:
            return False
        window = SortedList()

        for i in range(len(nums)):
            num = nums[i]

            idx = window.bisect_left(num - valueDiff)

            if idx < len(window) and window[idx] <= num + valueDiff:
                return True

            window.add(num)

            if i >= indexDiff:
                window.remove(nums[i - indexDiff])

        return False


if __name__ == '__main__':
    # print(Solution().containsNearbyAlmostDuplicate(nums = [1,2,3,1], indexDiff = 3, valueDiff = 0))
    print(Solution().containsNearbyAlmostDuplicate(nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3))
    # print(Solution().containsNearbyAlmostDuplicate([1,2,1,1], indexDiff = 1, valueDiff = 0))
    # print(Solution().containsNearbyAlmostDuplicate([1,2,5,6,7,2,4], indexDiff = 4, valueDiff = 0))