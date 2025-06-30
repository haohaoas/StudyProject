from typing import List


from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        cnt1 = sum(1 for x in nums1 if x == 0)
        cnt2 = sum(1 for x in nums2 if x == 0)
        if cnt1 == 0 and sum1 < sum2 + cnt2:
            sum2 += cnt2
            if sum1 == sum2:
                return sum1
            else:
                return -1

        if cnt2 == 0 and sum2 < sum1 + cnt1:
            sum1 += cnt1
            if sum1 == sum2:
                return sum1
            else:
                return -1
        return max(sum1 + cnt1, sum2 + cnt2)


if __name__ == '__main__':
    print(Solution().minSum(nums1 = [8,13,15,18,0,18,0,0,5,20,12,27,3,14,22,0], nums2 =[29,1,6,0,10,24,27,17,14,13,2,19,2,11]))