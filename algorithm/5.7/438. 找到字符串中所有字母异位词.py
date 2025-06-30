from collections import Counter
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        count=Counter(p)
        for i in range(len(s)-len(p)+1):
            if i==0:
                temp=Counter(s[:len(p)])
            else:
                temp[s[i-1]]-=1
                temp[s[i+len(p)-1]]+=1
            if temp==count:
                res.append(i)
        return res

if __name__ == '__main__':
    s = "cbaebabacd"
    p = "abc"
    print(Solution().findAnagrams(s = "abab", p = "ab"))