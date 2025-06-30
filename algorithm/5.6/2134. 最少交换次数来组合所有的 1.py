from collections import Counter, defaultdict
from math import inf
from typing import List


class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        counter=sum(nums)
        ans=inf
        min_nums=counter-sum(nums[:counter])
        for i in range(len(nums)):
            ans=min(ans,min_nums)
            if nums[i]==0:
                min_nums-=1
            if nums[(i+counter)%len(nums)]==0:
                min_nums+=1
        return ans


if __name__ == '__main__':
    nums=[1,1,1,0,0,1,0,1,1,0]
    s = Solution()
    print(s.minSwaps(nums))


