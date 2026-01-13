# class Grasp:
#     def __init__(self, name):
#         self.name = name
#
# a = Grasp('A')
# s = set([a])
# l = list(s)
#
# l[0].name = 'B'
# print(list(s)[0].name)  # 输出 'B'，说明 set 中的对象也被改了


import copy

class Node:
    def __init__(self, value):
        self.value = value

node = Node(42)
original_list = [node, node]

copied_list = copy.deepcopy(original_list)

print(copied_list[0] is copied_list[1])  # ✅ True：两个位置还是同一个新 node
print(copied_list[1] is node)            # ❌ False：已经不是原来的 node 了
print(original_list[1] is node)            # ❌ False：已经不是原来的 node 了