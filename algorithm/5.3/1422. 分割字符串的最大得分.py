from collections import Counter
class Solution:
    def maxScore(self, s: str) -> int:
        n = len(s)
        count=0
        for i in range(1,n):
            count_0=Counter(s[:i])
            count_1=Counter(s[i:])
            if count_0['0']+count_1['1']>count:
                count=count_0['0']+count_1['1']
        return count

if __name__ == '__main__':
    print(Solution().maxScore(s = "1111"))