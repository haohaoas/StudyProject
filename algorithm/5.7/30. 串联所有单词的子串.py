from collections import Counter
from typing import List


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        res=[]
        count=Counter(words)
        for i in range(len(s) - len(words) * len(words[0]) + 1):
            seen=[]
            for j in range(0, len(words)*len(words[0]),len(words[0])):
                seen.append(s[i+j:i+j+len(words[0])])
            if Counter(seen) == count:
                res.append(i)
        print(res)
if __name__ == '__main__':
    s = "barfoothefoobarman"
    words = ["foo","bar"]
    print(Solution().findSubstring(s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]))