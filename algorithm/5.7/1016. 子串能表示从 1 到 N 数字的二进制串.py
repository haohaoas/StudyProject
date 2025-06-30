class Solution:
    def queryString(self, s: str, n: int) -> bool:
        for i in range(n,n//2,-1):
            if bin(i)[2:] not in s:
                return False
        return True
if __name__ == '__main__':
    print(Solution().queryString(s = "0110", n=4))