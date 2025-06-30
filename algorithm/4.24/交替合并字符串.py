class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
            len1=len(word1)
            len2=len(word2)
            word3=""
            if len1>=len2:
                for i in range(len1):
                    if i>=len2:
                        word3+=word1[i]
                        continue
                    word3+=word1[i]+word2[i]
                return word3
            else:
                for i in range(len2):
                    if i>=len1:
                        word3+=word2[i]
                        continue
                    word3+=word1[i]+word2[i]
                return word3
if __name__ == '__main__':
    s = Solution()
    print(s.mergeAlternately("ab", "pqrs"))
