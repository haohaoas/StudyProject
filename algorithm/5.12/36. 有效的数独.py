from typing import List


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rampant=set()
        vertical=set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]!=".":
                    if (i,board[i][j]) in rampant or (board[i][j],j) in vertical:
                        return False
                    rampant.add((i,board[i][j]))
                    vertical.add((board[i][j],j))
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]!=".":
                    if (i//3,j//3,board[i][j]) in rampant or (board[i][j],i//3,j//3) in vertical:
                        return False
                    rampant.add((i//3,j//3,board[i][j]))
                    vertical.add((board[i][j],i//3,j//3))

        return True

if __name__ == '__main__':
    s = Solution()
    print(s.isValidSudoku([[".",".",".",".","5",".",".","1","."],[".","4",".","3",".",".",".",".","."],[".",".",".",".",".","3",".",".","1"],["8",".",".",".",".",".",".","2","."],[".",".","2",".","7",".",".",".","."],[".","1","5",".",".",".",".",".","."],[".",".",".",".",".","2",".",".","."],[".","2",".","9",".",".",".",".","."],[".",".","4",".",".",".",".",".","."]]))