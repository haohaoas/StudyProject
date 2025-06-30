from typing import List


class Solution:
    def kthSmallestProduct(self, nums1: List[int], nums2: List[int], k: int) -> int:
        pos=0
        num_list= []
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                pos+=1
                num_list.append(nums1[i]*nums2[j])
                if pos<k:
                    continue
                if pos==k:
                    num_list.sort()
                    return num_list[k-1]
                if pos>k:
                    break



if __name__ == '__main__':
    # print(Solution().kthSmallestProduct([1, 4, 7, 11, 15], [1, 3, 6, 10, 15], 1))
    print(Solution().kthSmallestProduct([-2,-1,0,1,2], [-3,-1,2,4,5], 3))