from typing import List


class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        return max(sum(i) for i in accounts)
if __name__ == '__main__':
    s = Solution()
    print(s.maximumWealth([[1, 2, 3], [3, 2, 1]]))
