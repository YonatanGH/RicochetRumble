# this file is taken from ex4, and we only took the parts that we needed
import sys
import inspect
import heapq
import random

class Pair(object):
    """
    A utility class to represent pairs (ordering of the objects in the pair does not matter).
    It is used to represent mutexes (for both actions and propositions)
    """

    def __init__(self, a, b):
        """
        Constructor
        """
        self.a = a
        self.b = b

    def __eq__(self, other):
        if (self.a == other.a) & (self.b == other.b):
            return True
        if (self.b == other.a) & (self.a == other.b):
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "(" + str(self.a) + "," + str(self.b) + ")"

    def __hash__(self):
        return hash(self.a) + hash(self.b)

class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.

    Note that this PriorityQueue does not allow you to change the priority
    of an item.  However, you may insert the same item multiple times with
    different priorities.
    """

    def __init__(self):
        self.heap = []
        self.init = False

    def push(self, item, priority):
        if not self.init:
            self.init = True
            try:
                item < item
            except:
                item.__class__.__lt__ = lambda x, y: True
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, priority_function):
        """priorityFunction (item) -> priority"""
        self.priorityFunction = priority_function  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer

    def push(self, item):
        """Adds an item to the queue with priority from the priority function"""
        PriorityQueue.push(self, item, self.priorityFunction(item))
