from collections import defaultdict


class Solution:
    def countLargestGroup(self, n: int) -> int:
        dp=defaultdict(int)
        count=0
        for i in range(1, n+1):
          dp_sum=sum(map(int,str(i)))
          dp[dp_sum]+=1
        for size in dp.values():
          if size==max(dp.values()):
            count+=1
        return count
if __name__ == '__main__':
    s= Solution()
    print(s.countLargestGroup(n = 13))