from itertools import accumulate
from typing import List


class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        if all(x < minK for x in nums) or all(x > maxK for x in nums):
            return 0
        elif len(set(nums)) == 1 and nums[0] == minK == maxK:
            n = len(nums)
            return n * (n + 1) // 2
        else:
            n = len(nums)
            last_min = last_max = last_out = -1  # 记录最近的minK、maxK和越界位置
            result = 0
            
            for i in range(n):
                if minK <= nums[i] <= maxK:
                    # 更新最近的minK和maxK位置
                    if nums[i] == minK:
                        last_min = i
                    if nums[i] == maxK:
                        last_max = i
                    
                    # 计算以当前位置结尾的有效子数组数量
                    if last_min != -1 and last_max != -1:
                        valid_start = min(last_min, last_max)
                        result += max(0, valid_start - last_out)
                else:
                    # 遇到越界数字，更新last_out
                    last_out = i
                    last_min = last_max = -1
            
            return result

if __name__ == '__main__':
    nums = [35054,398719,945315,945315,820417,945315,35054,945315,171832,945315,35054,109750,790964,441974,552913]
    minK = 35054
    maxK = 945315
    print(Solution().countSubarrays(nums, minK, maxK))
