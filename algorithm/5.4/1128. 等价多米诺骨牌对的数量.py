from collections import Counter
from typing import List


class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        count=Counter()
        result=0
        for a,b in dominoes:
           key=tuple(sorted([a,b]))
           print(key)
           result+=count[key]
           count[key]+=1
        return result
if __name__ == '__main__':
    dominoes=[[1,2],[1,2],[1,1],[1,2],[2,2]]
    print(Solution().numEquivDominoPairs(dominoes))