from typing import List
from collections import Counter


class Solution:
    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        n = len(tops)
        count_top = Counter(tops)
        count_bottom = Counter(bottoms)

        candidates = [count_top.most_common(1)[0][0], count_bottom.most_common(1)[0][0]]
        result = float('inf')

        for candidate in candidates:
            rotate_top = 0
            rotate_bottom = 0
            possible = True
            for i in range(n):
                if tops[i] != candidate and bottoms[i] != candidate:
                    possible = False
                    break
                elif tops[i] != candidate:
                    rotate_top += 1
                elif bottoms[i] != candidate:
                    rotate_bottom += 1
            if possible:
                result = min(result, rotate_top, rotate_bottom)

        return result if result != float('inf') else -1
