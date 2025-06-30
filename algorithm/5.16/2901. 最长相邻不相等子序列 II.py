from typing import List


class Solution:
    def getWordsInLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        def hamming_distance(s1, s2):
            if len(s1) != len(s2):
                return float('inf')
            return sum(a != b for a, b in zip(s1, s2))

        n = len(words)
        dp = [1] * n
        prev = [-1] * n
        for i in range(n):
            for j in range(i):
                if (
                        groups[j] != groups[i] and
                        len(words[j]) == len(words[i]) and
                        hamming_distance(words[j], words[i]) == 1
                ):
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

        # 回溯获得最长子序列的下标
        max_len = max(dp)
        idx = dp.index(max_len)
        path = []
        while idx != -1:
            path.append(words[idx])
            idx = prev[idx]
        return path[::-1]
if __name__ == '__main__':
    s=Solution()
    print(s.getWordsInLongestSubsequence(["baa","ada"],[1,2]))
    print(s.getWordsInLongestSubsequence(words = ["abbbb"], groups = [1]))
    print(s.getWordsInLongestSubsequence(words = ["bab","dab","cab"], groups = [1,2,2]))
    print(s.getWordsInLongestSubsequence(words = ["a","b","c","d"], groups = [1,2,3,4]))