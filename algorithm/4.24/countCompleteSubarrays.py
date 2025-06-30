from typing import List
from collections import Counter

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        nums_set = set(nums)
        nums_len = len(nums_set)
        count = 0
        left = 0
        num_counter = Counter()
        while left<len(nums):
            right=left
            num_counter.clear()
            while right<len(nums):
                num_counter[nums[right]]+=1
                if len(num_counter)==nums_len:
                    count+=1
                    break
                right+=1
            left+=1
        return count

if __name__ == '__main__':
    nums = [1,3,1,2,2]
    print(Solution().countCompleteSubarrays(nums))
