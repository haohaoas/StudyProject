class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        x, y, dx, dy = 0, 0, 0, 1
        for i in instructions:
            if i == 'L':
                dx, dy = -dy, dx
            elif i == 'R':
                dx, dy = dy, -dx
            elif i == 'G':
                x += dx
                y += dy
        return (x, y) == (0, 0) or (dx, dy) != (0, 1)

if __name__ == '__main__':
    print(Solution().isRobotBounded("GGLLGG"))