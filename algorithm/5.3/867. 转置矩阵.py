from typing import List


class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        return list(map(list, zip(*matrix)))

if __name__ == '__main__':
    s = Solution()
    print(s.transpose([[1,2,3],[4,5,6]]))