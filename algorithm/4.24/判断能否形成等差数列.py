from typing import List


class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
            arr.sort()
            for i in range(len(arr)):
                if i == 0:
                    continue
                if arr[i] - arr[i - 1] != arr[1] - arr[0]:
                    return False
            return True
if __name__ == '__main__':
    s = Solution()
    print(s.canMakeArithmeticProgression([3,5,1]))