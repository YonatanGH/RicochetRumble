# This file was taken from ex4, and we only took the parts that we needed
import util

# BEGIN SOLUTION
class Node:
    """AIMA: A node in a search tree. Contains a pointer 
    to the parent (the node that this is a successor of) 
    and to the actual state for this node. Note that if 
    a state is arrived at by two paths, then there are 
    two nodes with the same state.  Also includes the 
    action that got us to this state, and the total 
    path_cost (also known as g) to reach the node.  
    Other functions may add an f and h value; see 
    best_first_graph_search and astar_search for an 
    explanation of how the f and h values are handled. 
    You will not need to subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        if parent:
            self.path_cost = parent.path_cost + path_cost
            self.depth = parent.depth + 1
        else:
            self.path_cost = path_cost
            self.depth = 0

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def nodePath(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        result.reverse()
        return result

    def path(self):
        """
        Create a path of actions from the start to the current state
        """
        actions = []
        currnode = self
        while currnode.parent:
            actions.append(currnode.action)
            currnode = currnode.parent
        actions.reverse()
        return actions

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        return [Node(next, self, act, cost)
                for (next, act, cost) in problem.get_successors(self.state)]


REVERSE_PUSH = False


def graphSearch(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue. [Fig. 3.18]"""
    startstate = problem.get_start_state()
    fringe.push(Node(problem.get_start_state()))
    try:
        startstate.__hash__()
        visited = set()
    except:
        visited = list()

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.is_goal_state(node.state):
            return node.path()
        try:
            inVisited = node.state in visited
        except:
            visited = list(visited)
            inVisited = node.state in visited

        if not inVisited:
            if isinstance(visited, list):
                visited.append(node.state)
            else:
                visited.add(node.state)
            nextNodes = node.expand(problem)
            if REVERSE_PUSH: nextNodes.reverse()
            for nextnode in nextNodes:
                fringe.push(nextnode)
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    # BEGIN SOLUTION METHOD
    return graphSearch(problem,
                       util.PriorityQueueWithFunction(
                           lambda node: node.path_cost + heuristic(node.state, problem)))
    # END SOLUTION

    # BEGIN SOLUTION


# Abbreviations
astar = a_star_search

