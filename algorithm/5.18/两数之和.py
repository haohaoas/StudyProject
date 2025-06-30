import math
from typing import List, Optional


class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy=ListNode(0)
        current=dummy
        carry=0
        while l1 or l2 or carry:
            val1=l1.val if l1 else 0
            val2=l2.val if l2 else 0
            sum=val1+val2+carry
            carry=sum//10
            current.next=ListNode(sum%10)
            current=current.next
            l1=l1.next if l1 else None
            l2=l2.next if l2 else None
        return dummy.next



def build_list_node(param):
    dummy=ListNode(0)
    current=dummy
    for val in param:
        current.next = ListNode(val)
        current = current.next
    return dummy.next


if __name__ == '__main__':
    l1 = build_list_node([9,9,9,9,9,9,9])
    l2 = build_list_node([9,9,9,9])
    result = Solution().addTwoNumbers(l1, l2)

    # 打印结果链表
    while result:
        print(result.val, end=" -> ")
        result = result.next
    print("None")

