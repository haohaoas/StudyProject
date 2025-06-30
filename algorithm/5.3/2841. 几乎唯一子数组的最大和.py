from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        nums_list=nums[:k]
        max_sum_nums=0
        if len(set(nums_list))>=m:
            max_sum_nums=sum(nums_list)
        for i in range(k,len(nums)):
            nums_list.pop(0)
            nums_list.append(nums[i])
            if len(set(nums_list))>=m:
                max_sum_nums=max(max_sum_nums,sum(nums_list))
        return max_sum_nums

if __name__ == '__main__':
    print(Solution().maxSum([1,2,1,2,1,2,1], m = 3, k = 3))
