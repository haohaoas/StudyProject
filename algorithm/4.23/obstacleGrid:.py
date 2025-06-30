from typing import List


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m=len(obstacleGrid)
        n=len((obstacleGrid[0]))
        dp=[[0]*n for i in range(m)]
        dp[0][0]=1

if __name__ == '__main__':
    s=Solution()
    print(s.uniquePathsWithObstacles(obstacleGrid=[[0,0,0],[0,1,0],[0,0,0]]))