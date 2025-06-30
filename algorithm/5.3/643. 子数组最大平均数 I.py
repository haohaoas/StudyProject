from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        avg_nums=0.0
        sum_nums=0
        for i in range(k):
            sum_nums+=nums[i]
        avg_nums=sum_nums/k
        for i in range(k,len(nums)):
            if nums[i-k]==nums[i]:
                continue
            else:
                sum_nums=sum_nums-nums[i-k]+nums[i]
                avg_nums=max(avg_nums,sum_nums/k)
        return avg_nums
if __name__ == '__main__':
    print(Solution().findMaxAverage(nums = [1,12,-5,-6,50,3], k = 4))
