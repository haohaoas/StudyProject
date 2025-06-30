class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        concat = []
        if len(s1) + len(s2) != len(s3):
            return False
        for i in range(len(s1)*2):
            if i%2 == 0:
                concat.append(s2[i])
            else:
                concat.append(s1[i])
            if i>=len(s1):
                k=(i-len(s1))%2
                if k==0:
                    concat.append(s2[k])
                else:
                    concat.append(s1[k])
        return concat
if __name__ == '__main__':
    s1 = "aabcc"
    s2 = "dbbca"
    s3 = "aadbbcbcac"
    s= Solution()
    print(s.isInterleave(s1,s2,s3))