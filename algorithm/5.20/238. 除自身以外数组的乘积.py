from typing import List
from itertools import  accumulate

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        left=[1]*len(nums)
        for i in range(1,len(nums)):#  left[i]表示nums[0:i]的乘积
            left[i]=left[i-1]*nums[i-1]
        right=[1]*len(nums)
        for i in range(len(nums)-2,-1,-1):# right[i]表示nums[i+1:]的乘积
            right[i]=right[i+1]*nums[i+1]
            left[i]*=right[i]
        return left



if __name__ == '__main__':
    print(Solution().productExceptSelf([1,2,3,4]))