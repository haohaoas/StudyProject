from itertools import accumulate
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # 初始化 cnt 数组，用于存储前缀和取模的结果
        cnt = [0] * min(len(nums) + 1, modulo)
        cnt[0] = 1  # 初始化为空子数组的情况

        ans = s = 0  # ans 用于存储最终答案，s 用于记录满足条件的元素个数
        for x in nums:
            if x % modulo == k:  # 如果当前元素满足条件
                s += 1  # 更新 s

            if s >= k:  # 如果 s 大于等于 k，说明可能存在满足条件的子数组
                ans += cnt[(s - k) % modulo]  # 查找历史前缀和是否满足条件

            cnt[s % modulo] += 1  # 更新当前前缀和取模的结果

        return ans  # 返回最终答案

if __name__ == '__main__':
    nums = [1, 3, 5, 7, 9]  # 示例输入
    modulo = 7
    k = 1
    print(Solution().countInterestingSubarrays(nums, modulo, k))  # 输出结果
