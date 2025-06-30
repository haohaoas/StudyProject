from typing import List


class Solution:
    def calPoints(self, operations: List[str]) -> int:
        sum_list=[]
        if len(operations) == 0:
            return 0
        count=0
        for i in operations:
            if i == "+":
                sum_list.append(sum_list[-1]+sum_list[-2])
            elif i == "D":
                sum_list.append(sum_list[-1]*2)
            elif i == "C":
                sum_list.pop()
            else:
                sum_list.append(int(i))
        for i in sum_list:
            count += i
        return count
if __name__ == '__main__':
    operations = ["5","-2","4","C","D","9","+","+"]
    print(Solution().calPoints(operations))
