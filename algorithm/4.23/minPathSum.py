from functools import lru_cache
from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        #  递归解决
        def dfs(i, j):
            #越界检查
            if i>=m or j>=n:
                return float('inf')
            #  到达终点
            if i==m-1 and j==n-1:
                return grid[i][j]
            #  递归调用
            right=dfs(i,j+1)
            down=dfs(i+1,j)
            return grid[i][j]+min(right,down)
        m,n=len(grid), len(grid[0])
        return dfs(0,0)
    # 动态规划解决
    # def minPathSum(self, grid: List[List[int]]) -> int:
    #     m, n = len(grid), len(grid[0])
    #     dp = [[0] * n for _ in range(m)]
    #     dp[0][0]=grid[0][0]
    #     #左边的边界
    #     for i in range(1,n):
    #         dp[0][i]=dp[0][i-1]+grid[0][i]
    #     print(dp)
    #     #右边的边界
    #     for i in range(1,m):
    #         dp[i][0]=dp[i-1][0]+grid[i][0]
    #     print(dp)
    #     for i in range(1,m):
    #         for j in range(1,n):
    #             dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]
    #     return dp[m-1][n-1]
if __name__ == '__main__':
    grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    print(Solution().minPathSum(grid))
