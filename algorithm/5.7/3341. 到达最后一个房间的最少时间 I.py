from typing import List

from typing import List
import heapq

from collections import deque
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        visited = [[False] * m for _ in range(n)]
        queue = deque()
        queue.append((0, 0, 0))  # (x, y, current_time)

        while queue:
            i, j, t = queue.popleft()

            # 已访问过且当前时间更长，就不继续
            if visited[i][j]:
                continue
            visited[i][j] = True

            # 到达终点
            if i == n - 1 and j == m - 1:
                return t

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    next_time = t + 1  # 移动时间
                    interval = moveTime[ni][nj]
                    if interval != 0 and next_time % interval != 0:
                        wait = interval - (next_time % interval)
                        next_time += wait
                    queue.append((ni, nj, next_time))

        return -1  # 如果无法到达



if __name__ == '__main__':
    s = Solution()
    moveTime =[[94,79,62,27,69,84],[6,32,11,82,42,30]]
    print(s.minTimeToReach(moveTime))