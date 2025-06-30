import heapq
import random
from typing import List


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        left, right = 0, len(nums) - 1
        i = 0
        while i <= right:
            if nums[left] == 0:
                left += 1
                i+=1
            elif nums[right] == 2:
                nums[right], nums[left] = nums[left], nums[right]
                right -= 1
            else:
                i+=1

def quick_sort(nums):
    if len(nums)<=1:
        return nums
    pivot=nums[0]
    left=[num for num in nums[1:] if num<pivot]
    right=[num for num in nums[1:] if num>=pivot]
    return quick_sort(left)+[pivot]+quick_sort(right)


if __name__ == '__main__':
    s = Solution()
    s.sortColors([2,0,2,1,1,0])