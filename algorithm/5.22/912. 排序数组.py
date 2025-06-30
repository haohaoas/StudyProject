import heapq
from typing import List


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        heapq.heapify(nums)
        new_list=[]
        while nums:
            new_list.append(heapq.heappop(nums))
        return new_list
if __name__ == '__main__':
    nums=[5,2,3,1]
    print(Solution().sortArray(nums))