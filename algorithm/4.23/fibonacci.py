class Solution:
    def fib(self, n: int) -> int:

        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        if n == 0:
            return dp[0]
        if n == 1:
            return dp[1]

        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
if __name__ == '__main__':
    s = Solution()
    print(s.fib(0))