from collections import Counter
from itertools import pairwise
class Solution:

    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_len=len(s1)
        s2_len=len(s2)
        count=Counter(s1)
        count_windows=Counter()
        for i in range(s2_len-s1_len+1):
            if i==0:
                count_windows=Counter(s2[:s1_len])
            else:
                count_windows[s2[i-1]]-=1
                count_windows[s2[i+s1_len-1]]+=1
            if count_windows==count:
                return True
        return False

if __name__ == '__main__':
    s1 = "ab"
    s2 = "eidbaooo"
    print(Solution().checkInclusion(s1= "ab",s2 = "eidboaoo"))