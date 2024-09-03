from tanks import Tank, Bullet
import copy
from abc import ABC, abstractmethod
import heapq
from bullet import Bullet
import numpy as np



PLAN_DOMAIN_FILE = "plan_domain.txt"
PLAN_PROBLEM_FILE = "plan_problem.txt"

# Planning graph tank
class PGTank(Tank):
    # This is a tank that uses the planning graph algorithm to decide what to do
    def __init__(self, board, x, y, number):
        super().__init__(board, x, y, number)
        self.plan = []
        self.isplan_uptodate = False
        self.current_plan_index = 0
        self.did_move = False

    def generate_plan(self, domain_file, problem_file):
        # Generate a plan for the tank
        self.create_domain_file(domain_file, self.board)
        self.create_problem_file(problem_file, self.board)
        gp = GraphPlan(domain_file, problem_file)
        plan = gp.graph_plan()
        if (plan is None):
            self.isplan_uptodate = False
        else:
            self.plan = plan
            self.current_plan_index = 0
            self.isplan_uptodate = True

    def create_domain_file(self, domain_file_name, board):
        pass

    def create_problem_file(self, problem_file_name, board):
        pass


    def move(self, _):
        super(PGTank, self).move(_)
        self.generate_plan(PLAN_DOMAIN_FILE, PLAN_PROBLEM_FILE)

        if self.isplan_uptodate:
            assert self.current_plan_index == 0

        action = self.plan[self.current_plan_index]
        if action in self.board.get_legal_moves(self):
            self.board.move_tank(self, action)
            self.current_plan_index += 1
            self.did_move = True
            self.isplan_uptodate = False

    def shoot(self, _):
        super(PGTank, self).shoot(_)

        if self.isplan_uptodate:
            assert self.current_plan_index == 0

        if not self.did_move:
            action = self.plan[self.current_plan_index]
            if action in self.board.get_legal_shoots(self):
                self.board.shoot_bullet(self, action)
                self.current_plan_index += 1
                self.isplan_uptodate = False

        self.did_move = False

class GraphPlan:
    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file
        # TODO

    # temporary:
    def graph_plan(self):
        return ["MOVE_UP"] # TODO - change this lol
