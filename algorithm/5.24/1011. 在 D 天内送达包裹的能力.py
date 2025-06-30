import math
from functools import lru_cache
from typing import List


class Solution:
    # def canShip(self,weights, days, cap):
    #     n = len(weights)
    #     @lru_cache(None)
    #     def dfs(i, day, sum_weight):
    #         if day > days:
    #             return False
    #         if i == n:
    #             return True
    #         # 放今天
    #         if sum_weight + weights[i] <= cap:
    #             if dfs(i + 1, day, sum_weight + weights[i]):
    #                 return True
    #         # 放明天
    #         if dfs(i + 1, day + 1, weights[i]):
    #             return True
    #         return False
    #
    #     return dfs(0, 1, 0)
    # def shipWithinDays(self, weights: List[int], days: int) -> int:
    #     left, right = max(weights), sum(weights)
    #     for cap in range(left, right + 1):
    #         if self.canShip(weights, days, cap):
    #             return cap
    """
    dfs(0, 1, 0) # 从第0个包裹，第一天，今天还没装
  ├─ 放第0个包裹到今天（sum=2）→ dfs(1, 1, 2)
  |    ├─ 再放第1个到今天（2+3=5<=cap）→ dfs(2, 1, 5)
  |    |    ├─ 再放第2个到今天（5+4=9>cap，不行）
  |    |    └─ 第2个只能放明天（sum=4）→ dfs(3, 2, 4)
  |    |         └─ i==n，返回True
  |    └─ 第1个放明天（sum=3）→ dfs(2, 2, 3)
  |         ├─ 再放第2个到今天（3+4=7>cap，不行）
  |         └─ 第2个只能再放明天（天数超了，返回False）
  └─ 第0个放明天（sum=2）→ dfs(1, 2, 2)
       ├─ 第1个到今天（2+3=5）→ dfs(2, 2, 5)
       |    ├─ 第2个放今天（5+4=9>cap，不行）
       |    └─ 放明天（天数超了，False）
       └─ 第1个放明天（天数超了，False）
    """
    # def shipWithinDays(self, weights: List[int], days: int) -> int:
    #     n = len(weights)
    #     # 前缀和，方便O(1)求区间和
    #     prefix = [0] * (n + 1)
    #     for i in range(n):
    #         prefix[i + 1] = prefix[i] + weights[i]
    #     dp = [[math.inf] * (days + 1) for _ in range(n + 1)] #  dp[i][d] 表示前i个包裹，第d天，最短运载能力
    #     dp[0][0] = 0
    #     for i in range(1, n + 1):  # 枚举包裹个数
    #         for d in range(1, days + 1):  # 枚举天数
    #             for j in range(i):  # 枚举第d天的起点
    #                 # max(dp[j][d-1], 第d天实际的总运载量)
    #                 dp[i][d] = min(dp[i][d], max(dp[j][d - 1], prefix[i] - prefix[j]))
    #     return int(dp[n][days])

    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def dayHelp(load):
            day=1#最少需要1天
            size=0#当前天数的载货量
            for i in weights:
                size+=i
                if size>load:#超过载货量
                    day+=1 #切换到第二天
                    size=i #新的一天要把上一天没装下的装上
            return day#返回所需要的天数
        left, right = max(weights), sum(weights)
        while left<right:
            mid=(left+right)//2
            if dayHelp(mid)>days: #需要的天数大于days
                left=mid+1
            else: #需要的天数小于days
                right=mid
        return left
if __name__ == '__main__':
    print(Solution().shipWithinDays(weights = [1,2,3,4,5,6,7,8,9,10], days = 5))