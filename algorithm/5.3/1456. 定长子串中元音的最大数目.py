class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        count = 0
        vowels = set('aeiou')
        max_count = 0
        for i in range(k):
            if s[i] in vowels:
                count += 1
        max_count=count
        for i in range(k, len(s)):
            if s[i] in vowels:
                count += 1
            if s[i-k] in vowels:
                count -= 1
            max_count = max(max_count, count)
        return max_count
if __name__ == '__main__':
    s = "aeiou"
    k = 2
    print(Solution().maxVowels(s, k))