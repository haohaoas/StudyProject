class Solution:
    def judgeCircle(self, moves: str) -> bool:
        x,y=0,0
        for i in moves:
            if i=='U':
                y+=1
            elif i=='D':
                y-=1
            elif i=='L':
                x-=1
            elif i=='R':
                x+=1
        return  x==0 and y==0
if __name__ == '__main__':
    s = Solution()
    print(s.judgeCircle("UD"))