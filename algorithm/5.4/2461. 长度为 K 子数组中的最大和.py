from collections import defaultdict
from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        ans=s=0
        cnt=defaultdict(int)
        for i,x in enumerate(nums):
            s+=x
            cnt[i]+=1
            left=i-k+1
            if left<0:
                continue
            if len(cnt)==k:
                ans=max(ans,s)
            output=nums[left]
            s-=output
            cnt[output]-=1
            if cnt[output]==0:
                del cnt[output]

        return ans

if __name__ == '__main__':
    print(Solution().maximumSubarraySum([1,5,4,2,9,9,9], k = 3))