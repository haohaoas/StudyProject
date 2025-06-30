from typing import List


class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        n=len(mat)
        sum=0
        for i in range(n):
            for j in range(n):
                if i==j or i+j==n-1:
                    sum+=mat[i][j]
                else:
                    sum+=0
        return sum
if __name__ == '__main__':
    print(Solution().diagonalSum([[1,2,3],[4,5,6],[7,8,9]]))


