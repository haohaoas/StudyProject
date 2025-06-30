from typing import List


class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
       max_nus= max(nums)
       dp=[0]*(max_nus+1)
       for i in nums:
           dp[i]+=i
       for i in range(2,max_nus+1):
           dp[i]=max(dp[i-1],dp[i-2]+dp[i])
       return dp[max_nus]


if __name__ == '__main__':
    s = Solution()
    print(s.deleteAndEarn([2,2,3,3,3,4]))