from typing import List


class Solution:
    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        res = 0
        for i in range(left, right + 1):
            if words[i][0] in 'aeiou' and words[i][-1] in 'aeiou':
                res  += 1
        return res

if __name__ == '__main__':
    print(Solution().vowelStrings(words = ["hey","aeo","mu","ooo","artro"], left = 1, right = 4))
