from action_layer import ActionLayer
from util import Pair
from proposition_layer import PropositionLayer


class PlanGraphLevel(object):
    """
    A class for representing a level in the plan graph.
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
        """
        current_layer_actions = self.action_layer.get_actions()

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
        """
        current_layer_actions = self.action_layer.get_actions()
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
        """
        current_layer_propositions = self.proposition_layer.get_propositions()
        current_layer_mutex_actions = self.action_layer.get_mutex_actions()

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
        expand the graph
        """
        previous_proposition_layer = previous_layer.get_proposition_layer()
        previous_layer_mutex_proposition = previous_proposition_layer.get_mutex_props()

        # Update the action layer given the previous proposition layer
        self.update_action_layer(previous_proposition_layer)
        # Update the mutex set in the action layer given the mutex proposition from the previous layer
        self.update_mutex_actions(previous_layer_mutex_proposition)
        # Update the propositions in the current proposition layer given the current action layer
        self.update_proposition_layer()
        # Update the mutex propositions in the current proposition layer
        self.update_mutex_proposition()

    def expand_without_mutex(self, previous_layer):
        previous_layer_proposition = previous_layer.get_proposition_layer()
        # Update the action layer given the previous proposition layer
        self.update_action_layer(previous_layer_proposition)
        # Update the propositions in the current proposition layer given the current action layer
        self.update_proposition_layer()


def mutex_actions(a1, a2, mutex_props):
    """
    This function returns true if a1 and a2 are mutex actions.
    """
    if Pair(a1, a2) not in PlanGraphLevel.independent_actions:
        return True
    return have_competing_needs(a1, a2, mutex_props)


def have_competing_needs(a1, a2, mutex_props):
    for p1 in a1.get_pre():
        for p2 in a2.get_pre():
            if Pair(p1, p2) in mutex_props:
                return True
    return False


def mutex_propositions(prop1, prop2, mutex_actions_list):
    for a1 in prop1.get_producers():
        for a2 in prop2.get_producers():
            if Pair(a1, a2) not in mutex_actions_list:
                return False
    return True
