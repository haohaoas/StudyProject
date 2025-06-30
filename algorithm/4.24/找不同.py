class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        list_s = list(s)
        list_t = list(t)
        if len(list_s) == 0:
            return list_t[0]
        else:
            for i in list_s:
                list_t.remove(i)
            return list_t[0]

if __name__ == '__main__':
    s = Solution()
    print(s.findTheDifference("abcd", "abcde"))