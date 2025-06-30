from typing import List


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]

        # 如果起点是障碍物，直接返回0
        if obstacleGrid[0][0] == 1:
            return 0

        dp[0][0] = 1

        # 初始化第一列
        for i in range(1, m):
            if obstacleGrid[i][0] == 0 and dp[i - 1][0] == 1:  # 当前位置不是障碍物且上一行可以到达
                dp[i][0] = 1

        # 初始化第一行
        for j in range(1, n):
            if obstacleGrid[0][j] == 0 and dp[0][j - 1] == 1:  # 当前位置不是障碍物且左一列可以到达
                dp[0][j] = 1

        # 动态规划填充剩余部分
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:  # 当前位置不是障碍物
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]


if __name__ == '__main__':
    s = Solution()
    print(s.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))