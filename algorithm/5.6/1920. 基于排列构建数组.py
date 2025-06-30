from typing import List


class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        return [nums[nums[i]] for i in range(len(nums))]

if __name__ == '__main__':
    s = Solution()
    print(s.buildArray([0,2,1,5,3,4]))