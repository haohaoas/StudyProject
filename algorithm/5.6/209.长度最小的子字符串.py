import itertools
from math import inf
from typing import List
from itertools import combinations

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if sum(nums)<target:
            return 0
        left=0
        min_len=inf
        sum_num=0
        for right in range(len(nums)):
            sum_num+=nums[right]
            while sum_num>=target:
                min_len=min(min_len,right-left+1)
                sum_num-=nums[left]
                left+=1
        return min_len


if __name__ == '__main__':
    s = Solution()
    print(s.minSubArrayLen(target = 213, nums = [12,28,83,4,25,26,25,2,25,25,25,12]))