from typing import List
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        m, n = len(board), len(board[0])
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((i, j, board[i][j]))
        to_live = []
        to_die = []
        for i, j, val in cells:
            live_neighbors = 0
            for x, y in [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j),
                         (i + 1, j + 1)]:
                if 0 <= x < m and 0 <= y < n:
                    if board[x][y] == 1:
                        live_neighbors += 1
            if val == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    to_die.append((i, j))
            else:
                if live_neighbors == 3:
                    to_live.append((i, j))
        for i, j in to_die:
            board[i][j] = 0
        for i, j in to_live:
            board[i][j] = 1



if __name__ == '__main__':
    s = Solution()
    s.gameOfLife([[0,1,0],[0,0,1],[1,1,1],[0,0,0]])
    #[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]