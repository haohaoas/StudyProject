from collections import defaultdict


class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        sub_dict = defaultdict(int)
        for i, j in zip(range(len(s)), range(minSize, len(s) + 1)):
            sub = s[i:j]
            if len(set(sub)) <= maxLetters:
                sub_dict[sub] += 1
        return max(sub_dict.values()) if sub_dict else 0
if __name__ == '__main__':
    s = Solution()
    print(s.maxFreq("aabcabcab", maxLetters = 2, minSize = 2, maxSize = 3))