from typing import List

from sortedcontainers import SortedList


class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        nums.sort()
        max_len=0
        window=[]
        for i in range(len(nums)):
            window.append(nums[i])
            if len(window)==2:
                max_len=max(max_len,window[1]-window[0])
                window.pop(0)
        return max_len


if __name__ == '__main__':
    print(Solution().maximumGap([1,3,100]))
    print(Solution().maximumGap([1,100]))
    print(Solution().maximumGap([1,100,1000]))
    print(Solution().maximumGap([3,6,9,1]))
