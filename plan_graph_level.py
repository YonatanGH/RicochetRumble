from action_layer import ActionLayer
from util import Pair
from proposition import Proposition
from proposition_layer import PropositionLayer


class PlanGraphLevel(object):
    """
    A class for representing a level in the plan graph.
    For each level i, the PlanGraphLevel consists of the actionLayer and propositionLayer at this level in this order!
    """
    independent_actions = set()  # updated to the independent_actions of the problem (graph_plan.py line 32)
    actions = []  # updated to the actions of the problem (graph_plan.py line 33 and planning_problem.py line 36)
    props = []  # updated to the propositions of the problem (graph_plan.py line 34 and planning_problem.py line 36)

    @staticmethod
    def set_independent_actions(independent_actions):
        PlanGraphLevel.independent_actions = independent_actions

    @staticmethod
    def set_actions(actions):
        PlanGraphLevel.actions = actions

    @staticmethod
    def set_props(props):
        PlanGraphLevel.props = props

    def __init__(self):
        """
        Constructor
        """
        self.action_layer = ActionLayer()  # see action_layer.py
        self.proposition_layer = PropositionLayer()  # see proposition_layer.py

    def get_proposition_layer(self):  # returns the proposition layer
        return self.proposition_layer

    def set_proposition_layer(self, prop_layer):  # sets the proposition layer
        self.proposition_layer = prop_layer

    def get_action_layer(self):  # returns the action layer
        return self.action_layer

    def set_action_layer(self, action_layer):  # sets the action layer
        self.action_layer = action_layer

    def update_action_layer(self, previous_proposition_layer):
        """
        Updates the action layer given the previous proposition layer (see proposition_layer.py)
        You should add an action to the layer if its preconditions are in the previous propositions layer,
        and the preconditions are not pairwise mutex.
        all_actions is the set of all the action (include noOp) in the domain
        You might want to use those functions:
        previous_proposition_layer.is_mutex(prop1, prop2) returns true
        if prop1 and prop2 are mutex at the previous propositions layer
        previous_proposition_layer.all_preconds_in_layer(action) returns true
        if all the preconditions of action are in the previous propositions layer
        self.actionLayer.addAction(action) adds action to the current action layer
        """
        all_actions = PlanGraphLevel.actions

        "*** OUR CODE BELOW FOR Q4 ***"
        # An action is added to the action layer if its preconditions are in the previous proposition layer,
        # and the preconditions are not pairwise mutex.
        for action in all_actions:
            # Check if all preconditions are in the previous proposition layer
            if previous_proposition_layer.all_preconds_in_layer(action):
                # Check if the preconditions are not pairwise mutex
                preconditions = action.get_pre()
                # Flag to check if the preconditions are mutex
                is_mutex = False
                for p1 in preconditions:
                    for p2 in preconditions:
                        # Check if the preconditions are mutex
                        if previous_proposition_layer.is_mutex(p1, p2):
                            # If the preconditions are mutex, set is_mutex to True and check the next precondition
                            is_mutex = True
                            break
                    # If the preconditions are mutex, break the loop
                    if is_mutex:
                        break
                # If the preconditions are not mutex, add the action to the action layer
                if not is_mutex:
                    self.action_layer.add_action(action)

    def update_mutex_actions(self, previous_layer_mutex_proposition):
        """
        Updates the mutex set in self.action_layer,
        given the mutex proposition from the previous layer.
        current_layer_actions are the actions in the current action layer
        You might want to use this function:
        self.actionLayer.add_mutex_actions(action1, action2)
        adds the pair (action1, action2) to the mutex set in the current action layer
        Note that an action is *not* mutex with itself
        """
        current_layer_actions = self.action_layer.get_actions()

        "*** OUR CODE BELOW FOR Q5 ***"
        # Two actions are mutex if they have competing needs or inconsistent support.
        # Competing needs are actions that have preconditions that are mutex at the
        # proposition layer of the previous level.
        # Inconsistent support are actions that have conflicting effects.
        for a1 in current_layer_actions:
            for a2 in current_layer_actions:
                # An action is not mutex with itself
                if a1 != a2:
                    # Check if the actions are mutex
                    if mutex_actions(a1, a2, previous_layer_mutex_proposition):
                        # If the actions are mutex, add them to the mutex set in the current action layer
                        self.action_layer.add_mutex_actions(a1, a2)

    def update_proposition_layer(self):
        """
        Updates the propositions in the current proposition layer,
        given the current action layer.
        don't forget to update the producers list!
        Note that same proposition in different layers might have different producers lists,
        hence you should create two different instances.
        current_layer_actions is the set of all the actions in the current layer.
        You might want to use those functions:
        dict() creates a new dictionary that might help to keep track on the propositions that you've
               already added to the layer
        self.proposition_layer.add_proposition(prop) adds the proposition prop to the current layer

        """
        current_layer_actions = self.action_layer.get_actions()

        "*** OUR CODE BELOW FOR Q6 ***"
        # A proposition is added to the proposition layer if it is the effect of an action in the current action layer.
        # The producers list of the proposition is updated with the actions that have the proposition on their add list.
        # Create a dictionary to keep track of the propositions that have already been added to the layer
        propositions_dict = dict()
        for action in current_layer_actions:
            # Get the add list of the action
            add_list = action.get_add()
            for prop in add_list:
                # Check if the proposition has already been added to the layer
                if prop not in propositions_dict:
                    # If the proposition has not been added to the layer, add it to the layer
                    self.proposition_layer.add_proposition(prop)
                    # Update the producers list of the proposition with the action
                    prop.add_producer(action)
                    # Add the proposition to the dictionary to keep track of the propositions
                    # that have been added to the layer
                    propositions_dict[prop] = True
                else:
                    # If the proposition has already been added to the layer,
                    # update the producers list of the proposition with the action
                    prop.add_producer(action)

    def update_mutex_proposition(self):
        """
        updates the mutex propositions in the current proposition layer
        You might want to use those functions:
        mutex_propositions(prop1, prop2, current_layer_mutex_actions) returns true
        if prop1 and prop2 are mutex in the current layer
        self.proposition_layer.add_mutex_prop(prop1, prop2) adds the pair (prop1, prop2)
        to the mutex set of the current layer
        """
        current_layer_propositions = self.proposition_layer.get_propositions()
        current_layer_mutex_actions = self.action_layer.get_mutex_actions()

        "*** OUR CODE BELOW FOR Q7 ***"
        # Two propositions are mutex if all ways of achieving the propositions (that is, actions at the
        # same level) are pairwise mutex.
        for prop1 in current_layer_propositions:
            for prop2 in current_layer_propositions:
                # Check if the propositions are mutex
                if mutex_propositions(prop1, prop2, current_layer_mutex_actions):
                    # If the propositions are mutex, add them to the mutex set of the current layer
                    self.proposition_layer.add_mutex_prop(prop1, prop2)

    def expand(self, previous_layer):
        """
        Your algorithm should work as follows:
        First, given the propositions and the list of mutex propositions from the previous layer,
        set the actions in the action layer.
        Then, set the mutex action in the action layer.
        Finally, given all the actions in the current layer,
        set the propositions and their mutex relations in the proposition layer.
        """
        previous_proposition_layer = previous_layer.get_proposition_layer()
        previous_layer_mutex_proposition = previous_proposition_layer.get_mutex_props()

        "*** OUR CODE BELOW FOR Q8 ***"
        # Update the action layer given the previous proposition layer
        self.update_action_layer(previous_proposition_layer)
        # Update the mutex set in the action layer given the mutex proposition from the previous layer
        self.update_mutex_actions(previous_layer_mutex_proposition)
        # Update the propositions in the current proposition layer given the current action layer
        self.update_proposition_layer()
        # Update the mutex propositions in the current proposition layer
        self.update_mutex_proposition()

    def expand_without_mutex(self, previous_layer):
        """
        Questions 11 and 12
        You don't have to use this function
        """
        previous_layer_proposition = previous_layer.get_proposition_layer()

        "*** OUR CODE BELOW FOR Q11,Q12 ***"
        # Update the action layer given the previous proposition layer
        self.update_action_layer(previous_layer_proposition)
        # Update the propositions in the current proposition layer given the current action layer
        self.update_proposition_layer()


def mutex_actions(a1, a2, mutex_props):
    """
    This function returns true if a1 and a2 are mutex actions.
    We first check whether a1 and a2 are in PlanGraphLevel.independent_actions,
    this is the list of all the independent pair of actions (according to your implementation in question 1).
    If not, we check whether a1 and a2 have competing needs
    """
    if Pair(a1, a2) not in PlanGraphLevel.independent_actions:
        return True
    return have_competing_needs(a1, a2, mutex_props)


def have_competing_needs(a1, a2, mutex_props):
    """
    Complete code for deciding whether actions a1 and a2 have competing needs,
    given the mutex proposition from previous level (list of pairs of propositions).
    Hint: for propositions p  and q, the command  "Pair(p, q) in mutex_props"
          returns true if p and q are mutex in the previous level
    """

    "*** OUR CODE BELOW FOR Q2 ***"
    # according to the definition of competing needs, seen in the recitation:
    # Competing needs are actions that have preconditions that are mutex at the proposition layer of the previous level.
    for p1 in a1.get_pre():
        for p2 in a2.get_pre():
            if Pair(p1, p2) in mutex_props:
                return True
    return False


def mutex_propositions(prop1, prop2, mutex_actions_list):
    """
    complete code for deciding whether two propositions are mutex,
    given the mutex action from the current level (set of pairs of actions).
    Your update_mutex_proposition function should call this function
    You might want to use this function:
    prop1.get_producers() returns the set of all the possible actions in the layer that have prop1 on their add list
    """

    "*** OUR CODE BELOW FOR Q3 ***"
    # according to the definition of mutex, seen in the recitation:
    # Two propositions are mutex if all ways of achieving the propositions (that is, actions at the
    # same level) are pairwise mutex.
    for a1 in prop1.get_producers():
        for a2 in prop2.get_producers():
            if Pair(a1, a2) not in mutex_actions_list:
                return False
    return True
