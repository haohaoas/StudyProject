from typing import List


class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        def generator():
            counter = [0] * 1001
            for num in arr1:
                counter[num] += 1
            for num in arr2:
                for _ in range(counter[num]):
                    yield num
                counter[num] = 0
            for num in range(1001):
                if counter[num]:
                    for _ in range(counter[num]):
                        yield num

        return list(generator())

if __name__ == '__main__':
    arr1 = [2,3,1,3,2,4,6,7,9,2,19]
    arr2 = [2,1,4,3,9,6]
    print(Solution().relativeSortArray(arr1, arr2))