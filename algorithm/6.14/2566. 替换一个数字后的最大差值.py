class Solution:
    def minMaxDifference(self, num: int) -> int:
        s=str(num)
        max_s=''
        min_s=''
        for i in range(len(s)):
            if s[i]=='9':
                continue
            else:
                max_s = s.replace(s[i],'9')
                break
        if not max_s:
            max_s = s
        for j in range(len(s)):
            if s[j]=='0':
                continue
            else:
                min_s = s.replace(s[j],'0')
                break
        if not min_s:
            min_s = s
        return int(max_s)-int(min_s)


if __name__ == '__main__':
    print(Solution().minMaxDifference(99999))