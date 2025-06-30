from typing import List


class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
            board = [['' for _ in range(3)] for _ in range(3)]

            # 填充棋盘
            for i, (r, c) in enumerate(moves):
                board[r][c] = 'A' if i % 2 == 0 else 'B'

            # 检查胜利情况
            for player in ['A', 'B']:
                # 行
                for row in board:
                    if all(cell == player for cell in row):
                        return player
                # 列
                for col in range(3):
                    if all(board[row][col] == player for row in range(3)):
                        return player
                # 对角线
                if all(board[i][i] == player for i in range(3)) or \
                        all(board[i][2 - i] == player for i in range(3)):
                    return player

            return "Draw" if len(moves) == 9 else "Pending"
if __name__ == '__main__':
    moves = [[0,0],[1,1],[0,1],[0,2],[1,0],[2,0]]
    print(Solution().tictactoe(moves))
