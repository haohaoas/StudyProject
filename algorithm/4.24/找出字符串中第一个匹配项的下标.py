class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if haystack.find(needle) == -1:
            return -1
        else:
            return haystack.find(needle)
if __name__ == '__main__':
    s=Solution()
    print(s.strStr("leetcode","leeto"))