from typing import List


class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n=len(cardPoints)
        total_pst=sum(cardPoints)
        windows_size=n-k
        cur_pts=sum(cardPoints[:windows_size])
        min_sum=cur_pts
        for i in range(windows_size,n):
            cur_pts=cur_pts-cardPoints[i-windows_size]+cardPoints[i]
            min_sum=min(min_sum,cur_pts)
        return total_pst-min_sum
if __name__ == '__main__':
    print(Solution().maxScore([1,2,3,4,5,6,1], 3))