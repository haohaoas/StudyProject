from typing import List

import logging
from collections import Counter
from typing import List

class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        count = Counter(words)
        ans = 0
        central_used = False
        for word in list(count.keys()):
            rev = word[::-1]
            if word == rev:
                pairs = count[word] // 2
                ans += pairs * 4
                count[word] -= pairs * 2
                if not central_used and count[word] > 0:
                    ans += 2
                    central_used = True
            else:
                if rev in count:
                    pairs = min(count[word], count[rev])
                    ans += pairs * 4
                    count[word] -= pairs
                    count[rev] -= pairs

        return ans


if __name__ == '__main__':
    words = ["lc", "cl", "gg"]
    s = Solution()
    print(s.longestPalindrome(words))
    print(s.longestPalindrome(words = ["ab","ty","yt","lc","cl","ab"]))
