from typing import List
from operator import add

class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        list_arr=sum(arr[:k])
        count=0
        if list_arr//k >= threshold:
            count+=1
        for i in range(k,  len(arr)):
            list_arr = list_arr-arr[i-k]+arr[i]
            if  list_arr//k >= threshold:
                count+=1
        return count


if __name__ == '__main__':
    arr = [2,2,2,2,5,5,5,8]
    k = 3
    threshold = 4
    print(Solution().numOfSubarrays(arr, k, threshold))
