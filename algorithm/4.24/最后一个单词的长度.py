class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s=s.strip()
        return len(s.split(" ")[-1])
if __name__ == '__main__':
    s=Solution()
    print(s.lengthOfLastWord(s="Hello World"))