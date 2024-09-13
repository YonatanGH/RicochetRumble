from proposition_layer import PropositionLayer
from plan_graph_level import PlanGraphLevel
from pgparser import PgParser
from action import Action
from search import a_star_search

class PlanningProblem:
    def __init__(self, domain_file, problem_file):
        """
        Constructor
        """
        p = PgParser(domain_file, problem_file)
        self.actions, self.propositions = p.parse_actions_and_propositions()
        # list of all the actions and list of all the propositions

        initial_state, goal = p.parse_problem()
        # the initial state and the goal state are lists of propositions

        self.initialState = frozenset(initial_state)
        self.goal = frozenset(goal)

        self.create_noops()
        # creates noOps that are used to propagate existing propositions from one layer to the next

        PlanGraphLevel.set_actions(self.actions)
        PlanGraphLevel.set_props(self.propositions)
        self.expanded = 0

    def get_start_state(self):
        return self.initialState

    def is_goal_state(self, state):
        if self.goal_state_not_in_prop_layer(state):
            return False
        return True

    def get_successors(self, state):
        """
        get the successors of the given state
        """
        self.expanded += 1
        successors = []
        for action in self.actions:
            # if action is noop, continue
            if action.is_noop():
                continue
            if action.all_preconds_in_list(state):
                successor = set(state)
                successor.difference_update(set(action.get_delete()))
                successor.update(set(action.get_add()))
                successors.append((frozenset(successor), action, 1))
        return successors

    @staticmethod
    def get_cost_of_actions(actions):
        return len(actions)

    def goal_state_not_in_prop_layer(self, propositions):
        """
        Helper function that receives a list of propositions (propositions) and returns true
        if not all the goal propositions are in that list
        """
        for goal in self.goal:
            if goal not in propositions:
                return True
        return False

    def create_noops(self):
        """
        Creates the noOps that are used to propagate propositions from one layer to the next
        """
        for prop in self.propositions:
            name = prop.name
            precon = []
            add = []
            precon.append(prop)
            add.append(prop)
            delete = []
            act = Action(name, precon, add, delete, True)
            self.actions.append(act)


def max_level(state, planning_problem):
    """
    max level heuristic
    """
    # create a new proposition layer
    prop_layer_init = PropositionLayer()
    # update the proposition layer with the propositions of the state
    for prop in state:
        prop_layer_init.add_proposition(prop)
    # create a new plan graph level (level is the action layer and the propositions layer)
    pg_init = PlanGraphLevel()
    # update the new plan graph level with the proposition layer
    pg_init.set_proposition_layer(prop_layer_init)

    # go over all layers of propositions, and find the minimum level where all goal propositions are present
    graph = [pg_init]
    level = 0
    while not planning_problem.is_goal_state(graph[level].get_proposition_layer().get_propositions()):
        if is_fixed(graph, level):
            return float('inf')
        pg_next = PlanGraphLevel()
        pg_next.expand_without_mutex(graph[level])
        graph.append(pg_next)
        level += 1
    return level


def level_sum(state, planning_problem):
    """
    level sum heuristic
    """
    prop_layer_init = PropositionLayer()
    for prop in state:
        prop_layer_init.add_proposition(prop)
    pg_init = PlanGraphLevel()
    pg_init.set_proposition_layer(prop_layer_init)

    # go over all layers of propositions, and find the minimum level for each goal proposition
    graph = [pg_init]
    level = 0
    goals = list(planning_problem.goal)
    cur_sum = 0

    while not planning_problem.is_goal_state(graph[level].get_proposition_layer().get_propositions()):
        if is_fixed(graph, level):
            return float('inf')
        # check if any of the goals are in the current level
        for goal in goals:
            if goal in graph[level].get_proposition_layer().get_propositions():
                cur_sum += level
                goals.remove(goal)  # remove the goal from the list, as we found it already
        pg_next = PlanGraphLevel()
        pg_next.expand_without_mutex(graph[level])
        graph.append(pg_next)
        level += 1

    cur_sum += level * len(goals)  # add the remaining goals
    return cur_sum


def is_fixed(graph, level):
    """
    Checks if we have reached a fixed point, with no effect for expanding the plan graph
    """
    if level == 0:
        return False
    return len(graph[level].get_proposition_layer().get_propositions()) == len(
        graph[level - 1].get_proposition_layer().get_propositions())


def null_heuristic(*args, **kwargs):
    return 0


def solve(problem, heuristic=null_heuristic):
    return a_star_search(problem, heuristic)
