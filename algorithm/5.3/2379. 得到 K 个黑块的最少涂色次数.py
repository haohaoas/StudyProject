class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        cnt=blocks[:k].count('W')
        ans=cnt
        for i in range(k,len(blocks)):
            if blocks[i-k]=='W':
                cnt-=1
            if blocks[i]=='W':
                cnt+=1
            ans=min(ans,cnt)
        return ans

if __name__ == '__main__':
    blocks = "WBWBBBW"
    k = 2
    print(Solution().minimumRecolors(blocks, k))