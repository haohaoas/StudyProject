from math import sqrt


class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        return [ i for i in range(1, int(sqrt(num)) + 1) if i * i == num] != []
