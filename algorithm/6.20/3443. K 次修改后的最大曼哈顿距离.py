class Solution:
    # 'N'：向北移动 1 个单位。
    # 'S'：向南移动 1 个单位。
    # 'E'：向东移动 1 个单位。
    # 'W'：向西移动 1 个单位。
    def __init__(self):
        self.x=0
        self.y=0
    def result(self,x, y):
        return abs(x)+abs(y)

    def maxDistance(self, s: str, k: int) -> int:
        max_len=0
        positive=0
        negative=0
        for i, c in enumerate(s):
            if c=='N':
                positive+=1
            elif c=='S':
                negative+=1
            elif c=='E':
                positive+=1
            elif c=='W':
                negative+=1



        for i in range(len(s)):
            if s[i]=='N':
                if positive<=negative:
                    if k>0:
                        self.y-=1
                        k-=1
                        max_len = max(max_len, self.result(self.x, self.y))
                        continue
                self.y+=1
                max_len=max(max_len,self.result(self.x,self.y))
            elif s[i]=='S':
                if positive>negative:
                    if k > 0:
                        self.y+=1
                        k-=1
                        max_len = max(max_len, self.result(self.x, self.y))
                        continue
                self.y-=1
                max_len=max(max_len,self.result(self.x,self.y))
            elif s[i]=='E':
                if positive<=negative:
                    if k>0:
                        self.x-=1
                        k-=1
                        max_len = max(max_len, self.result(self.x, self.y))
                        continue
                self.x+=1
                max_len=max(max_len,self.result(self.x,self.y))
            elif s[i]=='W':
                if positive>=negative:
                    if k>0:
                        self.x+=1
                        k-=1
                        max_len = max(max_len, self.result(self.x, self.y))
                        continue
                self.x-=1
                max_len=max(max_len,self.result(self.x,self.y))

        return max_len






if __name__ == '__main__':
    # print(Solution().maxDistance(s = "NWSE", k =1))
    # print(Solution().maxDistance(s = "SN", k =0))
    print(Solution().maxDistance(s = "WEES", k =2))