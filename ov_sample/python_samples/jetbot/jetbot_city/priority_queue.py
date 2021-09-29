# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


class PriorityQueue(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        a = 0
        b = len(self.items) - 1
        while a <= b:
            c = a + (b - a) // 2
            if self.items[c][0] < item[0]:  # 0, 1 (0),
                a = c + 1
            elif self.items[c][0] > item[0]:
                b = c - 1
            else:
                break

        if a >= len(self.items):
            idx = len(self.items)
        elif b < 0:
            idx = 0
        else:
            idx = a + (b - a) // 2

        self.items.insert(idx, item)

    def pop(self):
        return self.items.pop(0)

    def empty(self):
        return len(self.items) == 0
