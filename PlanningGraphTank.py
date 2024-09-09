import tanks
from tanks import Tank, Bullet
from graph_plan import *
from planning_problem import *
import copy
from abc import ABC, abstractmethod
import heapq
from bullet import Bullet
import numpy as np



PLAN_DOMAIN_FILE = "plan_domain.txt"
PLAN_PROBLEM_FILE = "plan_problem.txt"

ACCOUNT_BULLET_MOVEMENT = False

# Planning graph tank
class PGTank(Tank):
    # This is a tank that uses the planning graph algorithm to decide what to do
    def __init__(self, board, x, y, number):
        super().__init__(board, x, y, number)
        self.plan = []
        self.isplan_uptodate = False
        self.current_plan_index = 0
        self.did_move = False
        self.num_turns_to_replan = 1 # how many turns to wait before replanning,
        # for best - do 1, for faster - do more than 1

    def generate_plan(self, domain_file, problem_file):
        # Generate a plan for the tank
        self.create_domain_file(domain_file, self.board)
        self.create_problem_file(problem_file, self.board)
        # gp = GraphPlan(domain_file, problem_file)
        # plan = gp.graph_plan()
        heuristics = [null_heuristic, max_level, level_sum]
        my_heuristic = heuristics[1]
        prob = PlanningProblem(domain_file, problem_file)
        plan = solve(prob, my_heuristic)

        if (plan is None):
            self.isplan_uptodate = False
        else:
            self.plan = plan
            self.current_plan_index = 0
            self.isplan_uptodate = True
            # remove the _from_{x}_{y} suffix from every action in the plan

            self.plan = [action.name.split("_from")[0] for action in self.plan]
            # print(self.plan)

    def create_domain_file(self, domain_file_name, board):
        # Create the domain file for the planning graph
        domain_file = open(domain_file_name, 'w')
        domain_file.write("Propositions:\n")
        # propositions: tank_at_x_y, enemy_at_x_y, bullet_at_x_y, wall_at_x_y, empty_at_x_y
        for x in range(-1, board.width+1):
            for y in range(-1, board.height+1):
                domain_file.write(f"tank_at_{x}_{y} ")
                domain_file.write(f"enemy_at_{x}_{y} ")
                domain_file.write(f"bullet_at_{x}_{y} ")
                domain_file.write(f"wall_at_{x}_{y} ")
                domain_file.write(f"empty_at_{x}_{y} ")
        domain_file.write("\nActions:\n")
        # actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP_LEFT, MOVE_UP_RIGHT, MOVE_DOWN_LEFT, MOVE_DOWN_RIGHT
        #          SHOOT_UP, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT, SHOOT_UP_LEFT, SHOOT_UP_RIGHT, SHOOT_DOWN_LEFT, SHOOT_DOWN_RIGHT

        # also, for each action, we need to add the preconditions and effects
        # TODO: and also, we need to add to the effects the movement of all bullets
        # in order to do that, we will gather two list: one of the current location of the bullets,
        # and one of the new location of the bullets, and then we will add the effects of the bullets
        # to the actions
        # problem with this: only considers the first step of the bullet, but this is the best I can think of
        bullets = board.bullets
        bullets_locations = [(bullet.x, bullet.y) for bullet in bullets]
        bullets_new_locations = [(bullet.x + tanks.str_to_vals[bullet.direction][0],
                                  bullet.y + tanks.str_to_vals[bullet.direction][1])
                                 for bullet in bullets]
        # now create a list of the proposition affected by the bullets
        bullets_remove = ["bullet_at_{x}_{y}" for x, y in bullets_locations]
        bullets_add = ["bullet_at_{x}_{y}" for x, y in bullets_new_locations]
        # now turn those to strings with a space between the elements
        bullets_remove = " ".join(bullets_remove)
        bullets_remove = " " + bullets_remove
        bullets_add = " ".join(bullets_add)
        bullets_add = " " + bullets_add
        if not ACCOUNT_BULLET_MOVEMENT:
            bullets_remove = ""
            bullets_add = ""


        for x in range(board.width):
            for y in range(board.height):
                legal_actions = self.get_legal_actions()
                for action in legal_actions:
                    if "MOVE" in action:
                        if "UP_LEFT" in action:
                            # if x == 0 or y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y-1}\n")
                            domain_file.write(f"add: tank_at_{x-1}_{y-1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x-1}_{y-1}\n")
                        elif "UP_RIGHT" in action:
                            # if x == board.width - 1 or y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y-1}\n")
                            domain_file.write(f"add: tank_at_{x+1}_{y-1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x+1}_{y-1}\n")
                        elif "DOWN_LEFT" in action:
                            # if x == 0 or y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y+1}\n")
                            domain_file.write(f"add: tank_at_{x-1}_{y+1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x-1}_{y+1}\n")
                        elif "DOWN_RIGHT" in action:
                            # if x == board.width - 1 or y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y+1}\n")
                            domain_file.write(f"add: tank_at_{x+1}_{y+1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x+1}_{y+1}
                        elif "UP" in action: # ~~~~ NOTE: the 4 direction must be after the diagonal directions
                            # if y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y-1}\n")
                            domain_file.write(f"add: tank_at_{x}_{y-1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x}_{y-1}
                        elif "DOWN" in action:
                            # if y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y+1}\n")
                            domain_file.write(f"add: tank_at_{x}_{y+1}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x}_{y+1}\n")
                        elif "LEFT" in action:
                            # if x == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y}\n")
                            domain_file.write(f"add: tank_at_{x-1}_{y}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x-1}_{y}\n")
                        elif "RIGHT" in action:
                            # if x == board.width - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y}\n")
                            domain_file.write(f"add: tank_at_{x+1}_{y}\n") # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n") # empty_at_{x+1}_{y}\n")
        for x in range(board.width):
            for y in range(board.height):
                legal_actions = self.get_legal_actions()
                for action in legal_actions:
                    # """

                    if "SHOOT" in action:
                        if "UP_LEFT" in action:
                            # if x == 0 or y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y-1}\n")
                            domain_file.write(f"add: bullet_at_{x-1}_{y-1}\n")
                        elif "UP_RIGHT" in action:
                            # if x == board.width - 1 or y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y-1}\n")
                            domain_file.write(f"add: bullet_at_{x+1}_{y-1}\n")
                        elif "DOWN_LEFT" in action:
                            # if x == 0 or y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y+1}\n")
                            domain_file.write(f"add: bullet_at_{x-1}_{y+1}\n")
                        elif "DOWN_RIGHT" in action:
                            # if x == board.width - 1 or y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y+1}\n")
                            domain_file.write(f"add: bullet_at_{x+1}_{y+1}\n")
                        elif "UP" in action:
                            # if y == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y-1}\n")
                            domain_file.write(f"add: bullet_at_{x}_{y-1}\n")
                        elif "DOWN" in action:
                            # if y == board.height - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y+1}\n")
                            domain_file.write(f"add: bullet_at_{x}_{y+1}\n")
                        elif "LEFT" in action:
                            # if x == 0:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x-1}_{y}\n")
                            domain_file.write(f"add: bullet_at_{x-1}_{y}\n")
                        elif "RIGHT" in action:
                            # if x == board.width - 1:
                            #     continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x+1}_{y}\n")
                            domain_file.write(f"add: bullet_at_{x+1}_{y}\n")

                    # """

        domain_file.close()

    def create_problem_file(self, problem_file_name, board):
        # Create the problem file for the planning graph
        problem_file = open(problem_file_name, 'w')
        problem_file.write("Initial state: ")
        my_tank_num = self.number
        enemy_tank_num = 1 if my_tank_num == 2 else 2
        for x in range(-1, board.width+1):
            for y in range(-1, board.height+1):
                if x==-1 or y==-1 or x==board.width or y==board.height:
                    problem_file.write(f"wall_at_{x}_{y} ")
                elif board.is_wall(x, y):
                    problem_file.write(f"wall_at_{x}_{y} ")
                elif board.tank1.x == x and board.tank1.y == y:
                    if my_tank_num == 1:
                        problem_file.write(f"tank_at_{x}_{y} ")
                        problem_file.write(f"empty_at_{x}_{y} ")
                    else:
                        problem_file.write(f"enemy_at_{x}_{y} ")
                        problem_file.write(f"empty_at_{x}_{y} ")
                elif board.tank2.x == x and board.tank2.y == y:
                    if my_tank_num == 2:
                        problem_file.write(f"tank_at_{x}_{y} ")
                    else:
                        problem_file.write(f"enemy_at_{x}_{y} ")
                        problem_file.write(f"empty_at_{x}_{y} ")
                elif board.is_bullet(x, y):
                    problem_file.write(f"bullet_at_{x}_{y} ")
                    problem_file.write(f"empty_at_{x}_{y} ")
                else:
                    problem_file.write(f"empty_at_{x}_{y} ")
        problem_file.write("\nGoal state: ")

        enemy_x = board.tank1.x if my_tank_num == 2 else board.tank2.x
        enemy_y = board.tank1.y if my_tank_num == 2 else board.tank2.y
        problem_file.write(f"bullet_at_{enemy_x}_{enemy_y} ")

        # for x in range(board.width):
        #     for y in range(board.height):
        #         if my_tank_num == 1:
        #             if board.tank2.x == x and board.tank2.y == y:
        #                 problem_file.write(f"enemy_at_{x}_{y} ")
        #                 problem_file.write(f"bullet_at_{x}_{y} ")
        #         else:
        #             if board.tank1.x == x and board.tank1.y == y:
        #                 problem_file.write(f"enemy_at_{x}_{y} ")
        #                 problem_file.write(f"bullet_at_{x}_{y} ")
        problem_file.close()


    def move(self, _):
        super(PGTank, self).move(_)

        # bad program design. The interface should be the same for all tanks, and the other interface
        # with the update() is better than this.
        # in that case, the update() will do the generate of plan and also call move() and shoot()

        if self.current_plan_index+1 >= self.num_turns_to_replan or len(self.plan) <= self.num_turns_to_replan:
            self.generate_plan(PLAN_DOMAIN_FILE, PLAN_PROBLEM_FILE)

        if self.isplan_uptodate:
            assert self.current_plan_index == 0

        # print(self.plan, "\n", self.current_plan_index, "\n", self.isplan_uptodate) #TODO: REMOVE

        action = self.plan[self.current_plan_index]
        if "MOVE" not in action:
            return False
        # get the direction of the action: "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"
        action_direction = action[action.index("_")+1:]
        if action in self.get_legal_actions():
            self.current_plan_index += 1
            self.did_move = True
            self.isplan_uptodate = False
            d_x, d_y = tanks.str_to_vals[action_direction.lower()]
            new_x, new_y = self.x + d_x, self.y + d_y
            return self.board.move_tank(self, new_x, new_y, self.number)

    def shoot(self, _):
        super(PGTank, self).shoot(_)

        if self.isplan_uptodate:
            assert self.current_plan_index == 0

        if not self.did_move:
            action = self.plan[self.current_plan_index]
            if "SHOOT" not in action:
                return False
            # get the direction of the action: "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"
            action_direction = action[action.index("_")+1:]
            if action in self.get_legal_actions():
                d_x, d_y = tanks.str_to_vals[action_direction.lower()]
                shot_x, shot_y = self.x + d_x, self.y + d_y
                self.shots -= 1
                self.current_plan_index += 1
                self.isplan_uptodate = False
                self.board.add_bullet(Bullet(self.board, shot_x, shot_y, action_direction))

        self.did_move = False
