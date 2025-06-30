from typing import List


class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        nums_list=[]
        count=0
        for i in range(len(nums)-2):
            nums_list.append(nums[i:i+3])
        for i in nums_list:
                if 2*(i[0]+i[2])==i[1]:
                    count+=1
        return count
if __name__ == '__main__':
    print(Solution().countSubarrays([-1,-2,0]))
