from typing import List


class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        nums1=[i for i in nums[:n]]
        nums2=[i for i in nums[n:]]
        res=[]
        for i in range(n):
            res.append(nums1[i])
            res.append(nums2[i])
        return res
if __name__ == '__main__':
    s = Solution()
    print(s.shuffle([2,5,1,3,4,7], 3))
