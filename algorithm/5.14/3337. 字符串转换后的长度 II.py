from typing import List
from typing import List


class Solution:
    def lengthAfterTransformations(self, s: str, t: int, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        for _ in range(t):
            new_count = [0] * 26
            for i in range(len(nums)):
                if count[i] == 0:
                    continue
                cnt = count[i]
                count1 = nums[i]
                for j in range(1, count1 + 1):
                    char_idx = (i + j) % 26
                    new_count[char_idx] = (new_count[char_idx] + cnt) % MOD
            count = new_count

        return sum(count) % MOD
if __name__ == '__main__':
    s = "abcyy"
    t = 2
    nums = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]
    print(Solution().lengthAfterTransformations(s, t, nums))