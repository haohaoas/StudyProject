class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s + s).find(s, 1) != len(s)

if __name__ == '__main__':
    s = Solution()
    print(s.repeatedSubstringPattern(s="abab"))