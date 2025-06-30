from typing import List


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        fisrt_row=any(matrix[0][j]==0 for j in range(len(matrix[0])))
        fisrt_col=any(matrix[i][0]==0 for i in range(len(matrix)))
        for i in range(1,len(matrix)):
            for j in range(1,len(matrix[0])):
                if matrix[i][j]==0:
                    matrix[i][0]=matrix[0][j]=0
        for i in range(1,len(matrix)):
            for j in range(1,len(matrix[0])):
                if matrix[i][0]==0 or matrix[0][j]==0:
                    matrix[i][j]=0
        for i in range(len(matrix)):
            if fisrt_col:
                matrix[i][0]=0
        for j in range(len(matrix[0])):
            if fisrt_row:
                matrix[0][j]=0
        print(matrix)
if __name__ == '__main__':
    s = Solution()
    matrix=[[1,1,1],[1,0,1],[1,1,1]]
    s.setZeroes(matrix)