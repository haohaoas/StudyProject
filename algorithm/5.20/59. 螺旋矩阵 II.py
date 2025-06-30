from typing import List


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        list_matrix=[[0 for i in range(n)] for i in range(n)]
        left=0
        right=n
        top=0
        bottom=n
        num=1
        while left<=right and top<=bottom:
            for i in range(left,right):
                list_matrix[top][i]=num
                num+=1
            top+=1
            for i in range(top,bottom):
                list_matrix[i][right-1]=num
                num+=1
            right-=1
            if top<bottom:
                for i in range(right-1,left-1,-1):
                    list_matrix[bottom-1][i]=num
                    num+=1
                bottom-=1
            if left<right:
                for i in range(bottom-1,top-1,-1):
                    list_matrix[i][left]=num
                    num+=1
                left+=1
        return list_matrix
if __name__ == '__main__':
    print(Solution().generateMatrix(2))