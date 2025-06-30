from typing import List


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        str_num=[str (i) for i in digits]
        num=int(''.join(str_num))
        num+=1
        num=str(num)
        str_num=[int(i) for i in num]
        return str_num
if __name__ == '__main__':
    digits=[1,2,3]
    print(Solution().plusOne(digits))
