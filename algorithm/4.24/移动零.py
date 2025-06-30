from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        count=0
        for i in range(len(nums)):
            if nums[i] == 0:
                count += 1
        for i in range(count):
            nums.remove(0)
        for i in range(count):
            nums.append(0)
        print(nums)
if __name__ == '__main__':
    nums = [0, 0,1]
    Solution().moveZeroes(nums)