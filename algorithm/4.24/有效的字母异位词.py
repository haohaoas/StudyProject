from collections import defaultdict


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dc=defaultdict(int)
        dc1=defaultdict(int)
        if len(s)!=len(t):
            return False
        for i in s:
            dc[i]+=1
        for i in t:
            dc1[i]+=1
        if dc==dc1:
            return True
        else:
            return False
if __name__ == '__main__':
    s =Solution()
    print(s.isAnagram("anagrm","nagaram"))