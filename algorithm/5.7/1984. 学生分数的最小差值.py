from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans=max(nums[:k])-min(nums[:k])
        for i in range(k,len(nums)):
            ans=min(ans,max(nums[i-k+1:i+1])-min(nums[i-k+1:i+1]))
        return ans
if __name__ == '__main__':
    print(Solution().minimumDifference([87063,61094,44530,21297,95857,93551,9918],6))
