from typing import List


class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        res=[-1] * len(nums)
        window_sum = sum(nums[0:2 * k + 1])
        res[k] = window_sum // (2*k+1)
        for i in range(k + 1, len(nums) - k):
            window_sum = window_sum - nums[i - k - 1] + nums[i + k]
            res[i] = window_sum // (2*k+1)
        return res
if __name__ == '__main__':
    nums = [7,4,3,9,1,8,5,2,6]
    k = 3
    print(Solution().getAverages(nums, k))
