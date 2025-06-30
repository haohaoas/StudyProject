from collections import defaultdict
from typing import List


class Solution:

    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        total=0
        for i in range(len(grumpy)):
            if grumpy[i]==0:
                total+=customers[i]
        excal=0
        for i in range(minutes):
            if grumpy[i]==1:
                excal+=customers[i]
        max_customers=excal
        for i in range(minutes, len(grumpy)):
            if grumpy[i]==1:
                excal+=customers[i]
            if grumpy[i-minutes]==1:
                excal-=customers[i-minutes]
            max_customers=max(max_customers, excal)
        return total+max_customers


if __name__ == '__main__':
    s = Solution()
    # customers = [1, 0, 1, 2, 1, 1, 7, 5]
    # grumpy = [0, 1, 0, 1, 0, 1, 0, 1]
    # minutes = 3
    # customers = [1]
    # grumpy = [0]
    # minutes = 1
    customers =[4,10,10]
    grumpy = [1,1,0]
    minutes = 2
    print(s.maxSatisfied(customers, grumpy, minutes))