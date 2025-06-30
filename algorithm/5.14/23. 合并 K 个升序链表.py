import heapq
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        heap=[]
        for node in lists:
            while node:
                heapq.heappush(heap, node.val)
                node=node.next
        dummy=ListNode(0)
        current=dummy
        while heap:
            current.next = ListNode(heapq.heappop(heap))
            current = current.next
        return dummy.next


def build_list(param):
    head = ListNode(param[0])
    current = head
    for i in range(1, len(param)):
        current.next = ListNode(param[i])
        current = current.next
    return head


if __name__ == '__main__':
    s = Solution()
    list_input = [build_list([1,4,5]), build_list([1,3,4]), build_list([2,6])]
    merged = s.mergeKLists(list_input)
    while merged:
        print(merged.val)
        merged = merged.next
