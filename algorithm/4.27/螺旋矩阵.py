from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        matrix_list = []
        top, left = 0, 0
        bottom, right = len(matrix), len(matrix[0])
        
        while len(matrix_list) < len(matrix) * len(matrix[0]):
            for i in range(left, right):
                matrix_list.append(matrix[top][i])
            top += 1
            for i in range(top, bottom):
                matrix_list.append(matrix[i][right-1])
            right -= 1
            if top < bottom:
                for i in range(right-1, left-1, -1):
                    matrix_list.append(matrix[bottom-1][i])
                bottom -= 1
            if left < right:
                for i in range(bottom-1, top-1, -1):
                    matrix_list.append(matrix[i][left])
                left += 1
        
        return matrix_list

if __name__ == '__main__':
    print(Solution().spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))