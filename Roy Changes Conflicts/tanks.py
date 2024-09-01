import copy
from abc import ABC, abstractmethod
import heapq
from bullet import Bullet
from game_state import GameState
import numpy as np
import random

BEGINNING_SHOTS = 3
MAX_SHOTS = 5

ACTIONS = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP_LEFT', 'MOVE_UP_RIGHT', 'MOVE_DOWN_LEFT',
           'MOVE_DOWN_RIGHT',
           'SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP_LEFT', 'SHOOT_UP_RIGHT', 'SHOOT_DOWN_LEFT',
           'SHOOT_DOWN_RIGHT']


class Tank(ABC):
    def __init__(self, board, x, y, number):
        """
        Initialize a tank.
        
        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        self.board = board  # Reference to the board
        self.x = x  # X coordinate
        self.y = y  # Y coordinate
        self.shots = BEGINNING_SHOTS  # Shot counter
        self.number = number  # Tank number
        self.bullets = []  # List of bullets
        board.place_tank(self, number)

    def add_bullet(self):
        """ Add a bullet to the tank's shot counter. """
        self.shots += 1 if self.shots < MAX_SHOTS else 0

    @abstractmethod
    def move(self, direction):
        """Move the tank in a specified direction."""
        self.add_bullet()  # add one bullet each time the tank moves
        pass

    @abstractmethod
    def shoot(self, direction):
        """Shoot a bullet in a specified direction."""
        pass

    def __get_legal_actions_shoot(self):
        legal_actions = []
        if self.y > 0:
            legal_actions.append('SHOOT_UP')
        if self.y < self.board.size - 1:
            legal_actions.append('SHOOT_DOWN')
        if self.x > 0:
            legal_actions.append('SHOOT_LEFT')
        if self.x < self.board.size - 1:
            legal_actions.append('SHOOT_RIGHT')
        if self.y > 0 and self.x > 0:
            legal_actions.append('SHOOT_UP_LEFT')
        if self.y > 0 and self.x < self.board.size - 1:
            legal_actions.append('SHOOT_UP_RIGHT')
        if self.y < self.board.size - 1 and self.x > 0:
            legal_actions.append('SHOOT_DOWN_LEFT')
        if self.y < self.board.size - 1 and self.x < self.board.size - 1:
            legal_actions.append('SHOOT_DOWN_RIGHT')
        return legal_actions

    def __get_legal_actions_move(self):
        legal_actions = []
        if self.y > 0:
            legal_actions.append('MOVE_UP')
        if self.y < self.board.size - 1:
            legal_actions.append('MOVE_DOWN')
        if self.x > 0:
            legal_actions.append('MOVE_LEFT')
        if self.x < self.board.size - 1:
            legal_actions.append('MOVE_RIGHT')
        if self.y > 0 and self.x > 0:
            legal_actions.append('MOVE_UP_LEFT')
        if self.y > 0 and self.x < self.board.size - 1:
            legal_actions.append('MOVE_UP_RIGHT')
        if self.y < self.board.size - 1 and self.x > 0:
            legal_actions.append('MOVE_DOWN_LEFT')
        if self.y < self.board.size - 1 and self.x < self.board.size - 1:
            legal_actions.append('MOVE_DOWN_RIGHT')
        return legal_actions

    def get_legal_actions(self):
        """
        Get the legal actions for the tank.

        :return: List of legal actions.
        """
        legal_actions = []
        if self.shots > 0:
            legal_actions += self.__get_legal_actions_shoot()
        legal_actions += self.__get_legal_actions_move()
        return legal_actions

    def state_to_move(self, state):
        """
        Get the move that corresponds to the given state.

        :param state: The state to convert.
        :return: The move that corresponds to the given state.
        """
        pass

class PlayerTank(Tank):
    def __init__(self, board, x, y, number):
        """
        Initialize a player-controlled tank.
        
        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)

    def move(self, direction):
        """
        Move the tank in a specified direction.
        
        :param direction: Direction to move ('up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right').
        :return: True if move is valid, False otherwise.
        """
        super().move(direction)
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-1, -1),
            'up_right': (1, -1),
            'down_left': (-1, 1),
            'down_right': (1, 1)
        }
        if direction in directions:
            dx, dy = directions[direction]
            new_x, new_y = self.x + dx, self.y + dy
            return self.board.move_tank(self, new_x, new_y, self.number)
        return False

    def shoot(self, direction):
        """
        Shoot a bullet in a specified direction.
        
        :param direction: Direction to shoot ('up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right').
        """
        if self.shots > 0:
            directions = {
                'up': (0, -1),
                'down': (0, 1),
                'left': (-1, 0),
                'right': (1, 0),
                'up_left': (-1, -1),
                'up_right': (1, -1),
                'down_left': (-1, 1),
                'down_right': (1, 1)
            }
            # convert direction to lowercase
            # direction = direction.lower()
            # direction = direction[6:]
            if direction in directions:
                dx, dy = directions[direction]
                bullet = Bullet(self.board, self.x + dx, self.y + dy, direction)
                can_add = self.board.add_bullet(bullet)
                if can_add:
                    self.shots -= 1
                return can_add
        else:
            self.board.show_message("You can't shoot yet!")
            return False

class AStarTank(Tank):
    def __init__(self, board, x, y, number):
        """
        Initialize an AI-controlled tank using the A* algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)
        self.turns = 0  # Turn counter

    def set_data(self, x, y, number):
        """
        Set the tank's position.

        :param x: X coordinate.
        :param y: Y coordinate.
        """
        self.x = x
        self.y = y
        self.number = number

    def a_star_path(self, start, goal):
        """
        Compute the A* path from start to goal.

        :param start: Starting position (x, y).
        :param goal: Goal position (x, y).
        :return: List of positions (x, y) in the path.
        """

        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_list:
            current = heapq.heappop(open_list)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.board.size and 0 <= neighbor[1] < self.board.size:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def move(self, _):
        """
        Move the tank using A* algorithm to reach the goal.

        :param _: Unused parameter. Needed to match the parent class signature.
        :return: True if move is valid, False otherwise.
        """
        super(AStarTank, self).move(_)
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1

        tank_position = (target_tank.x, target_tank.y)
        path = self.a_star_path((self.x, self.y), tank_position)
        if path:
            next_step = path[0]
            return self.board.move_tank(self, next_step[0], next_step[1], self.number)
        return False

    def shoot(self, _):
        """
        Shoot a bullet at the target tank if within range.

        :param _: Unused parameter. Needed to match the parent class signature.
        """
        if self.shots > 0:
            if self.number == 1:
                target_tank = self.board.tank2
            else:
                target_tank = self.board.tank1
            tank_position = (target_tank.x, target_tank.y)
            if abs(self.x - tank_position[0]) <= 1 and abs(self.y - tank_position[1]) <= 1:
                direction_map = {
                    (0, -1): 'up',
                    (0, 1): 'down',
                    (-1, 0): 'left',
                    (1, 0): 'right',
                    (-1, -1): 'up_left',
                    (1, -1): 'up_right',
                    (-1, 1): 'down_left',
                    (1, 1): 'down_right'
                }
                dx, dy = tank_position[0] - self.x, tank_position[1] - self.y
                direction = direction_map.get((dx, dy))
                if direction:
                    self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, direction))
                    self.shots -= 1

    def state_to_move(self, state):
        """
        Convert the state to a move.

        :param state: State of the game.
        :return: Move to make.
        """
        if self.number == 1:
            target_tank = self.board.tank2
            tank_position = (state[2], state[3])
        else:
            target_tank = self.board.tank1
            tank_position = (state[0], state[1])
        direction_map = {
            (0, -1): 'SHOOT_UP',
            (0, 1): 'SHOOT_DOWN',
            (-1, 0): 'SHOOT_LEFT',
            (1, 0): 'SHOOT_RIGHT',
            (-1, -1): 'SHOOT_UP_LEFT',
            (1, -1): 'SHOOT_UP_RIGHT',
            (-1, 1): 'SHOOT_DOWN_LEFT',
            (1, 1): 'SHOOT_DOWN_RIGHT'
        }
        if abs(self.x - tank_position[0]) <= 1 and abs(self.y - tank_position[1]) <= 1:
            dx, dy = tank_position[0] - self.x, tank_position[1] - self.y
            return direction_map.get((dx, dy))
        else:
            path = self.a_star_path((self.x, self.y), tank_position)
            next_step = path[0]
            dx, dy = next_step[0] - self.x, next_step[1] - self.y
            return direction_map.get((dx, dy))

    def update(self):
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        if abs(self.x - target_tank.x) <= 1 and abs(self.y - target_tank.y) <= 1:
            self.shoot(None)
        else:
            self.move(None)

class QLearningTank(Tank):
    def __init__(self, board, x, y, number, lr=0.1, ds=0.95, er=0.1, ed=0.01, epochs=100):
        """
        Initialize an AI-controlled tank using the Q-learning algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)
        self.q_table = {}
        self.learning_rate = lr
        self.discount_factor = ds
        self.exploration_rate = er
        self.exploration_decay = ed
        self.epochs = epochs

        # self.fill_q_table()

    def set_data(self, x, y, number):
        """
        Set the tank's position.

        :param x: X coordinate.
        :param y: Y coordinate.
        """
        self.x = x
        self.y = y
        self.number = number

    def state_legal_actions(self, state, prefix):
        """
        Get the legal actions for a given state.

        :param state: Current state.
        :return: List of legal actions.
        """
        legal_actions = []
        for action in ACTIONS:
            if action == 'MOVE_UP' or (action == 'SHOOT_UP' and state[4] > 0):
                if 0 < state[1+prefix] < self.board.size:
                    legal_actions.append(action)
            elif action == 'MOVE_DOWN' or (action == 'SHOOT_DOWN' and state[4] > 0):
                if 0 <= state[1+prefix] < self.board.size - 1:
                    legal_actions.append(action)
            elif action == 'MOVE_LEFT' or (action == 'SHOOT_LEFT' and state[4] > 0):
                if 0 < state[0+prefix] < self.board.size:
                    legal_actions.append(action)
            elif action == 'MOVE_RIGHT' or (action == 'SHOOT_RIGHT' and state[4] > 0):
                if 0 <= state[0+prefix] < self.board.size - 1:
                    legal_actions.append(action)
            elif action == 'MOVE_UP_LEFT' or (action == 'SHOOT_UP_LEFT' and state[2+prefix] > 0):
                if 0 < state[0+prefix] < self.board.size and 0 < state[1+prefix] < self.board.size:
                    legal_actions.append(action)
            elif action == 'MOVE_UP_RIGHT' or (action == 'SHOOT_UP_RIGHT' and state[2+prefix] > 0):
                if 0 <= state[0+prefix] < self.board.size - 1 and 0 < state[1+prefix] < self.board.size:
                    legal_actions.append(action)
            elif action == 'MOVE_DOWN_LEFT' or (action == 'SHOOT_DOWN_LEFT' and state[2+prefix] > 0):
                if 0 < state[0+prefix] < self.board.size - 1 and 0 <= state[1+prefix] < self.board.size - 1:
                    legal_actions.append(action)
            elif action == 'MOVE_DOWN_RIGHT' or (action == 'SHOOT_DOWN_RIGHT' and state[2+prefix] > 0):
                if 0 <= state[0+prefix] < self.board.size - 1 and 0 <= state[1+prefix] < self.board.size - 1:
                    legal_actions.append(action)
        return legal_actions

    def fill_q_table(self):
        """
        Fill the Q-table with the given state.

        :param state: Current state.
        """

        def is_terminal_state(state):
            """
            Check if the given state is terminal.

            :param state: Current state.
            :return: True if terminal, False otherwise.
            """
            # check if there is a bullet in the same position as a tank
            if len(state) <= 6:
                return False
            i = 6
            while i < len(state):
                if state[i] == state[0] and state[i + 1] == state[1]:
                    return True
                if state[i] == state[3] and state[i + 1] == state[4]:
                    return True
                i += 5
            return False

        for i in range(self.epochs):
            state = self.get_random_state()
            while not is_terminal_state(state):
                legal_actions = self.state_legal_actions(state, 0)
                if np.random.rand() < self.exploration_rate:
                    action = np.random.choice(legal_actions)
                else:
                    q_values = [self.get_q_value(tuple(state), action) for action in legal_actions]
                    action = legal_actions[np.argmax(q_values)]
                reward = self.reward(state, action)
                next_state = self.next_state(list(state), action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.exploration_rate *= (1 - self.exploration_decay)

    def get_random_state(self):
        """
        Get a random state for the Q-table.

        :return: Random state.
        """
        tank1 = (random.randint(0, self.board.size - 1), random.randint(0, self.board.size - 1))
        tank2 = (random.randint(0, self.board.size - 1), random.randint(0, self.board.size - 1))
        shots1 = random.randint(0, MAX_SHOTS)
        shots2 = random.randint(0, MAX_SHOTS)
        return [tank1[0], tank1[1], shots1, tank2[0], tank2[1], shots2]

    def get_state(self):
        """
        Get the current state of the tank.

        :return: Tuple representing the state.
        """
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        state = [self.x, self.y, self.shots, target_tank.x, target_tank.y, target_tank.shots]
        for bullet in self.board.bullets:
            state.extend([bullet.x, bullet.y, bullet.direction, bullet.bounces, bullet.moves])
        return state

    def next_state_helper(self, state, action, prefix):
        """
        Get the next state after taking an action.

        :param state: Current state.
        :param action: Action taken.
        :return: Next state.
        """
        next_state = state.copy()
        if action == 'MOVE_UP':
            next_state[1+prefix] -= 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_DOWN':
            next_state[1+prefix] += 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_LEFT':
            next_state[0+prefix] -= 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_RIGHT':
            next_state[0+prefix] += 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_UP_LEFT':
            next_state[0+prefix] -= 1
            next_state[1+prefix] -= 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_UP_RIGHT':
            next_state[0+prefix] += 1
            next_state[1+prefix] -= 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_DOWN_LEFT':
            next_state[0+prefix] -= 1
            next_state[1+prefix] += 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'MOVE_DOWN_RIGHT':
            next_state[0+prefix] += 1
            next_state[1+prefix] += 1
            next_state[2+prefix] = min(MAX_SHOTS, next_state[2+prefix] + 1)
        elif action == 'SHOOT_UP':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix], next_state[1+prefix] - 1, 'up', 0, 0])
        elif action == 'SHOOT_DOWN':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix], next_state[1+prefix] + 1, 'down', 0, 0])
        elif action == 'SHOOT_LEFT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] - 1, next_state[1+prefix], 'left', 0, 0])
        elif action == 'SHOOT_RIGHT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] + 1, next_state[1+prefix], 'right', 0, 0])
        elif action == 'SHOOT_UP_LEFT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] - 1, next_state[1+prefix] - 1, 'up_left', 0, 0])
        elif action == 'SHOOT_UP_RIGHT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] + 1, next_state[1+prefix] - 1, 'up_right', 0, 0])
        elif action == 'SHOOT_DOWN_LEFT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] - 1, next_state[1+prefix] + 1, 'down_left', 0, 0])
        elif action == 'SHOOT_DOWN_RIGHT':
            next_state[2+prefix] = max(0, next_state[2+prefix] - 1)
            next_state.extend([next_state[0+prefix] + 1, next_state[1+prefix] + 1, 'down_right', 0, 0])
        i = 6
        while i < len(next_state):
            dx, dy = str_to_vals[next_state[i+2]]
            next_state[i] += dx
            next_state[i+1] += dy
            # Handle wall bounces
            if next_state[i] < 0 or next_state[i] >= self.board.size:
                dx = -dx  # Bounce off vertical walls
                next_state[i] += 2 * dx
                next_state[i+2] = vals_to_str[-dx, dy]
                next_state[i+3] += 1
                if next_state[i+3] >= 3:
                    next_state = next_state[:i] + next_state[i+5:]
                    continue
            if next_state[i+1] < 0 or next_state[i+1] >= self.board.size:
                dy = -dy  # Bounce off horizontal walls
                next_state[i+1] += 2 * dy
                next_state[i+2] = vals_to_str[dx, -dy]
                next_state[i+3] += 1
                if next_state[i+3] >= 3:
                    next_state = next_state[:i] + next_state[i+5:]
                    continue
            next_state[i+4] += 1
            if next_state[i+4] >= 10:
                next_state = next_state[:i] + next_state[i+5:]
            i += 5
        return next_state

    def next_state(self, state, action):
        """
        Get the next state after taking an action.

        :param state: Current state.
        :param action: Action taken.
        :return: Next state.
        """
        next_state = self.next_state_helper(state, action, 0)
        next_state = self.next_state_helper(next_state, self.state_to_move(next_state), 3)
        return tuple(next_state)

    def state_to_move(self, state):
        """
        Get the move corresponding to a state.

        :param state: State to convert.
        :return: Move corresponding to the state.
        """
        q_values = [self.get_q_value(tuple(state), action) for action in self.state_legal_actions(state, 3)]
        action = self.state_legal_actions(state, 3)[np.argmax(q_values)]
        return action

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        :param state: Current state.
        :param action: Action taken.
        :return: Q-value for the state-action pair.
        """
        return self.q_table.get((state, action), self.reward(state, action))

    def choose_action(self, state):
        """
        Choose the best action for a given state.

        :param state: Current state.
        :return: Best action for the state.
        """
        # if np.random.rand() < self.exploration_rate: TODO is exploration in use after filling the table?
        #     return np.random.choice(self.state_legal_actions(state))
        q_values = [self.get_q_value(tuple(state), action) for action in self.state_legal_actions(state, 0)]
        action = self.state_legal_actions(state, 0)[np.argmax(q_values)]
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning algorithm.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state.
        """
        state = tuple(state)
        q_value = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, next_action) for next_action in ACTIONS]
        max_q_value = np.max(next_q_values)
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_q_value - q_value)
        self.q_table[(state, action)] = new_q_value

    def calculate_shoot_reward(self, state, action):
        def chebyshev_distance(pos1, pos2):
            """Calculate the Chebyshev distance between two points."""
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

        def count_bullets_in_area(opponent_pos):
            """
            Counts the number of bullets within a 3x3 area around the opponent.
            """

            bullet_positions = []
            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                i += 5

            x_min = max(opponent_pos[0] - 1, 0)
            x_max = min(opponent_pos[0] + 1, self.board.size - 1)
            y_min = max(opponent_pos[1] - 1, 0)
            y_max = min(opponent_pos[1] + 1, self.board.size - 1)

            bullets_in_area = 0

            for bullet_pos in bullet_positions:
                if x_min <= bullet_pos[0] <= x_max and y_min <= bullet_pos[1] <= y_max:
                    bullets_in_area += 1

            return bullets_in_area

        if state[2] == 0:
            return float('-inf') # illegal move

        # Maximum possible distance (considering a maximum Manhattan distance with bounces)
        max_possible_distance = 2 * (self.board.size - 1)

        # Step through each position on the trajectory
        x, y = state[0], state[1]
        if action == 'SHOOT_UP' or action == 'up':
            dx, dy = 0, -1
        elif action == 'SHOOT_DOWN' or action == 'down':
            dx, dy = 0, 1
        elif action == 'SHOOT_LEFT' or action == 'left':
            dx, dy = -1, 0
        elif action == 'SHOOT_RIGHT' or action == 'right':
            dx, dy = 1, 0
        elif action == 'SHOOT_UP_LEFT' or action == 'up_left':
            dx, dy = -1, -1
        elif action == 'SHOOT_UP_RIGHT' or action == 'up_right':
            dx, dy = 1, -1
        elif action == 'SHOOT_DOWN_LEFT' or action == 'down_left':
            dx, dy = -1, 1
        elif action == 'SHOOT_DOWN_RIGHT' or action == 'down_right':
            dx, dy = 1, 1
        else:
            return float('-inf') # should never be reached

        if chebyshev_distance((state[0], state[1]), (state[3], state[4])) == 1:
            if chebyshev_distance((state[0] + dx, state[1] + dy), (state[3], state[4])) == 0:
                return 1000
            return -1000

        # check if there is a bullet aimed at the tank
        bullet_positions = []
        bullet_directions = []

        i = 6
        while i < len(state):
            bullet_positions.append((state[i], state[i + 1]))
            bullet_directions.append(state[i + 2])
            i += 5

        for i in range(len(bullet_directions)):
            if bullet_directions[i] == 'up':
                dx, dy = 0, -1
            elif bullet_directions[i] == 'down':
                dx, dy = 0, 1
            elif bullet_directions[i] == 'left':
                dx, dy = -1, 0
            elif bullet_directions[i] == 'right':
                dx, dy = 1, 0
            elif bullet_directions[i] == 'up_left':
                dx, dy = -1, -1
            elif bullet_directions[i] == 'up_right':
                dx, dy = 1, -1
            elif bullet_directions[i] == 'down_left':
                dx, dy = -1, 1
            elif bullet_directions[i] == 'down_right':
                dx, dy = 1, 1
            else:
                return float('-inf')
            bullet_directions[i] = (dx, dy)

        avoidance_score = 0
        for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
            # Predict the next position of the bullet
            next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])
            # Check if the bullet is aimed at the new position
            # if next_bullet_pos == (state[3], state[4]):
            #     return -1000
            if next_bullet_pos == (state[0], state[1]):
                return -1000

            # handle bullet bounces
            if next_bullet_pos[0] < 0 or next_bullet_pos[0] >= self.board.size:
                bullet_dir = (-bullet_dir[0], bullet_dir[1])
            if next_bullet_pos[1] < 0 or next_bullet_pos[1] >= self.board.size:
                bullet_dir = (bullet_dir[0], -bullet_dir[1])

            next_bullet_pos = (next_bullet_pos[0] + bullet_dir[0], next_bullet_pos[1] + bullet_dir[1])
            # Check if the bullet is aimed at the new position
            if next_bullet_pos == (state[0], state[1]):
                return -1000

        min_distance = float('inf')

        for _ in range(10):
            # Calculate Chebyshev distance from current position on trajectory to the opponent
            distance_to_opponent = chebyshev_distance((x + dx, y + dy), (state[3], state[4]))
            min_distance = min(min_distance, distance_to_opponent)

            # Move to the next point on the trajectory
            x += dx
            y += dy

            # Handle wall bounces
            if x < 0 or x >= self.board.size:
                dx = -dx  # Bounce off vertical walls
                x += 2 * dx
            if y < 0 or y >= self.board.size:
                dy = -dy  # Bounce off horizontal walls
                y += 2 * dy

        # Proximity score for the best point (minimum product of distances)
        proximity_score = 1 / np.exp(min_distance / (self.board.size - 2)) - 0.35  # Exponential decrease in score

        # Calculate escape difficulty based on the number of bullets in a 3x3 area around the opponent
        bullets_in_area = count_bullets_in_area((state[3], state[4]))
        escape_difficulty_score = 1 - np.exp(-bullets_in_area)  # Exponential increase in difficulty

        # Exponential penalty for ammo (less ammo results in a lower score)
        ammo_penalty_factor = np.exp(state[2]) / np.exp(MAX_SHOTS)

        # Combine scores with weighted factors
        score = (proximity_score * 0.6) + (escape_difficulty_score * 0.3) + (ammo_penalty_factor * 0.1)

        return score

    def calculate_move_reward(self, state, action):
        def chebyshev_distance(pos1, pos2):
            """Calculate the Chebyshev distance between two points."""
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

        def bullet_avoidance_score(new_pos):
            """
            Calculate the bullet avoidance score based on bullets aimed at the new position
            and their distances.
            """
            bullet_positions = []
            bullet_directions = []

            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                bullet_directions.append(state[i + 2])
                i += 5

            for i in range(len(bullet_directions)):
                if bullet_directions[i] == 'up':
                    dx, dy = 0, -1
                elif bullet_directions[i] == 'down':
                    dx, dy = 0, 1
                elif bullet_directions[i] == 'left':
                    dx, dy = -1, 0
                elif bullet_directions[i] == 'right':
                    dx, dy = 1, 0
                elif bullet_directions[i] == 'up_left':
                    dx, dy = -1, -1
                elif bullet_directions[i] == 'up_right':
                    dx, dy = 1, -1
                elif bullet_directions[i] == 'down_left':
                    dx, dy = -1, 1
                elif bullet_directions[i] == 'down_right':
                    dx, dy = 1, 1
                else:
                    return float('-inf')
                bullet_directions[i] = (dx, dy)

            avoidance_score = 0
            for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
                # Predict the next position of the bullet
                next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])

                # Check if the bullet is aimed at the new position
                if next_bullet_pos == new_pos:
                    return -500

                distance_to_bullet = chebyshev_distance(new_pos, bullet_pos)
                avoidance_score -= np.exp(-distance_to_bullet)  # Exponential penalty for proximity

            return np.exp(avoidance_score)

        # Ammo management score (higher reward for lower ammo count)=
        ammo_reward_factor = 1 - (state[2] / MAX_SHOTS)  # Linearly scale from 1 (min ammo) to 0 (max ammo)

        # calculate the new position after the move
        x, y = state[0], state[1]
        if action == 'MOVE_UP' or action == 'up':
            y -= 1
        elif action == 'MOVE_DOWN' or action == 'down':
            y += 1
        elif action == 'MOVE_LEFT' or action == 'left':
            x -= 1
        elif action == 'MOVE_RIGHT' or action == 'right':
            x += 1
        elif action == 'MOVE_UP_LEFT' or action == 'up_left':
            x -= 1
            y -= 1
        elif action == 'MOVE_UP_RIGHT' or action == 'up_right':
            x += 1
            y -= 1
        elif action == 'MOVE_DOWN_LEFT' or action == 'down_left':
            x -= 1
            y += 1
        elif action == 'MOVE_DOWN_RIGHT' or action == 'down_right':
            x += 1
            y += 1

        # Calculate proximity score to the opponent
        distance_to_opponent_old = chebyshev_distance((state[0], state[1]), (state[3], state[4]))
        distance_to_opponent_new = chebyshev_distance((x, y), (state[3], state[4]))

        # Reward for being at Chebyshev distance 3, penalize for less or more
        if distance_to_opponent_new < 2:
            return -1000
        else:
            proximity_score = (distance_to_opponent_old / distance_to_opponent_new) - 1

        # Bullet avoidance score
        avoidance_score = bullet_avoidance_score((x, y)) - bullet_avoidance_score((state[0], state[1]))

        # Combine scores with weighted factors
        avoidance_coef = 0.3 * len(self.board.bullets)
        other_coef = 0.3 * MAX_SHOTS - len(self.board.bullets)
        score = (ammo_reward_factor * (0.3 + other_coef)) + (proximity_score * (0.4 + other_coef)) + (
                    avoidance_score * avoidance_coef)

        return score

    def reward(self, state, action):
        """
        Get the reward for a given state-action pair.

        :param state: Current state.
        :param action: Action taken.
        :return: Reward for the state-action pair.
        """
        if action.startswith('SHOOT'):
            return self.calculate_shoot_reward(state, action)
        return self.calculate_move_reward(state, action)

    def move(self, action):
        """
        Move the tank using the Q-learning algorithm to reach the goal.

        :param _: Unused parameter. Needed to match the parent class signature.
        :return: True if move is valid, False otherwise.
        """
        super(QLearningTank, self).move(action)

        state = self.get_state()
        next_state = self.next_state(state, action)
        self.update_q_table(self.get_state(), action, self.reward(self.get_state(), action), next_state)
        return self.board.move_tank(self, next_state[0], next_state[1], self.number)

    def shoot(self, action):
        """
        Shoot a bullet at the target tank if within range.

        :param _: Unused parameter. Needed to match the parent class signature.
        """
        super(QLearningTank, self).shoot(action)

        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-1, -1),
            'up_right': (1, -1),
            'down_left': (-1, 1),
            'down_right': (1, 1)
        }

        state = self.get_state()
        next_state = self.next_state(state, action)
        if action == 'SHOOT_UP':
            dx, dy = 0, -1
        elif action == 'SHOOT_DOWN':
            dx, dy = 0, 1
        elif action == 'SHOOT_LEFT':
            dx, dy = -1, 0
        elif action == 'SHOOT_RIGHT':
            dx, dy = 1, 0
        elif action == 'SHOOT_UP_LEFT':
            dx, dy = -1, -1
        elif action == 'SHOOT_UP_RIGHT':
            dx, dy = 1, -1
        elif action == 'SHOOT_DOWN_LEFT':
            dx, dy = -1, 1
        elif action == 'SHOOT_DOWN_RIGHT':
            dx, dy = 1, 1
        else:
            return False
        action = action[6:]
        action = action.lower()
        bullet = Bullet(self.board, self.x + dx, self.y + dy, action)
        can_add = self.board.add_bullet(bullet)
        self.bullets.append(bullet)
        if can_add:
            self.shots -= 1
            self.update_q_table(self.get_state(), action, self.reward(self.get_state(), action), next_state)
        return can_add

    def update(self):
        """
        Update the tank's state.
        """
        action = self.choose_action(self.get_state())
        if action.startswith('MOVE'):
            self.move(action)
        elif action.startswith('SHOOT'):
            self.shoot(action)
        # for action_loop in self.state_legal_actions(self.get_state()):
        #     print(action_loop, end=' ')
        #     print(self.reward(self.get_state(), action_loop), end=' ')
        # print(action)

class MinimaxTank(Tank):
    def __init__(self, board, x, y, number):
        """
        Initialize an AI-controlled tank using the Minimax algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)
        self.depth = 2 # TODO change to 3

    def calculate_shoot_reward(self, state, action):
        def chebyshev_distance(pos1, pos2):
            """Calculate the Chebyshev distance between two points."""
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

        def count_bullets_in_area(opponent_pos):
            """
            Counts the number of bullets within a 3x3 area around the opponent.
            """

            bullet_positions = []
            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                i += 5

            x_min = max(opponent_pos[0] - 1, 0)
            x_max = min(opponent_pos[0] + 1, self.board.size - 1)
            y_min = max(opponent_pos[1] - 1, 0)
            y_max = min(opponent_pos[1] + 1, self.board.size - 1)

            bullets_in_area = 0

            for bullet_pos in bullet_positions:
                if x_min <= bullet_pos[0] <= x_max and y_min <= bullet_pos[1] <= y_max:
                    bullets_in_area += 1

            return bullets_in_area

        if state[2] == 0:
            return float('-inf')  # illegal move

        # Maximum possible distance (considering a maximum Manhattan distance with bounces)
        max_possible_distance = 2 * (self.board.size - 1)

        # Step through each position on the trajectory
        x, y = state[0], state[1]
        if action == 'SHOOT_UP' or action == 'up':
            dx, dy = 0, -1
        elif action == 'SHOOT_DOWN' or action == 'down':
            dx, dy = 0, 1
        elif action == 'SHOOT_LEFT' or action == 'left':
            dx, dy = -1, 0
        elif action == 'SHOOT_RIGHT' or action == 'right':
            dx, dy = 1, 0
        elif action == 'SHOOT_UP_LEFT' or action == 'up_left':
            dx, dy = -1, -1
        elif action == 'SHOOT_UP_RIGHT' or action == 'up_right':
            dx, dy = 1, -1
        elif action == 'SHOOT_DOWN_LEFT' or action == 'down_left':
            dx, dy = -1, 1
        elif action == 'SHOOT_DOWN_RIGHT' or action == 'down_right':
            dx, dy = 1, 1
        else:
            return float('-inf')  # should never be reached

        if chebyshev_distance((state[0], state[1]), (state[3], state[4])) == 1:
            if chebyshev_distance((state[0] + dx, state[1] + dy), (state[3], state[4])) == 0:
                return 1000
            return -1000

        # check if there is a bullet aimed at the tank
        bullet_positions = []
        bullet_directions = []

        i = 6
        while i < len(state):
            bullet_positions.append((state[i], state[i + 1]))
            bullet_directions.append(state[i + 2])
            i += 5

        for i in range(len(bullet_directions)):
            if bullet_directions[i] == 'up':
                dx, dy = 0, -1
            elif bullet_directions[i] == 'down':
                dx, dy = 0, 1
            elif bullet_directions[i] == 'left':
                dx, dy = -1, 0
            elif bullet_directions[i] == 'right':
                dx, dy = 1, 0
            elif bullet_directions[i] == 'up_left':
                dx, dy = -1, -1
            elif bullet_directions[i] == 'up_right':
                dx, dy = 1, -1
            elif bullet_directions[i] == 'down_left':
                dx, dy = -1, 1
            elif bullet_directions[i] == 'down_right':
                dx, dy = 1, 1
            else:
                return float('-inf')
            bullet_directions[i] = (dx, dy)

        avoidance_score = 0
        for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
            # Predict the next position of the bullet
            next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])
            # Check if the bullet is aimed at the new position
            # if next_bullet_pos == (state[3], state[4]):
            #     return -1000
            if next_bullet_pos == (state[0], state[1]):
                return -1000

            # handle bullet bounces
            if next_bullet_pos[0] < 0 or next_bullet_pos[0] >= self.board.size:
                bullet_dir = (-bullet_dir[0], bullet_dir[1])
            if next_bullet_pos[1] < 0 or next_bullet_pos[1] >= self.board.size:
                bullet_dir = (bullet_dir[0], -bullet_dir[1])

            next_bullet_pos = (next_bullet_pos[0] + bullet_dir[0], next_bullet_pos[1] + bullet_dir[1])
            # Check if the bullet is aimed at the new position
            if next_bullet_pos == (state[0], state[1]):
                return -1000

        min_distance = float('inf')

        for _ in range(10):
            # Calculate Chebyshev distance from current position on trajectory to the opponent
            distance_to_opponent = chebyshev_distance((x + dx, y + dy), (state[3], state[4]))
            min_distance = min(min_distance, distance_to_opponent)

            # Move to the next point on the trajectory
            x += dx
            y += dy

            # Handle wall bounces
            if x < 0 or x >= self.board.size:
                dx = -dx  # Bounce off vertical walls
                x += 2 * dx
            if y < 0 or y >= self.board.size:
                dy = -dy  # Bounce off horizontal walls
                y += 2 * dy

        # Proximity score for the best point (minimum product of distances)
        proximity_score = 1 / np.exp(min_distance / (self.board.size - 2)) - 0.35  # Exponential decrease in score

        # Calculate escape difficulty based on the number of bullets in a 3x3 area around the opponent
        bullets_in_area = count_bullets_in_area((state[3], state[4]))
        escape_difficulty_score = 1 - np.exp(-bullets_in_area)  # Exponential increase in difficulty

        # Exponential penalty for ammo (less ammo results in a lower score)
        ammo_penalty_factor = np.exp(state[2]) / np.exp(MAX_SHOTS)

        # Combine scores with weighted factors
        score = (proximity_score * 0.6) + (escape_difficulty_score * 0.3) + (ammo_penalty_factor * 0.1)

        return score

    def calculate_move_reward(self, state, action):
        def chebyshev_distance(pos1, pos2):
            """Calculate the Chebyshev distance between two points."""
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

        def bullet_avoidance_score(new_pos):
            """
            Calculate the bullet avoidance score based on bullets aimed at the new position
            and their distances.
            """
            bullet_positions = []
            bullet_directions = []

            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                bullet_directions.append(state[i + 2])
                i += 5

            for i in range(len(bullet_directions)):
                if bullet_directions[i] == 'up':
                    dx, dy = 0, -1
                elif bullet_directions[i] == 'down':
                    dx, dy = 0, 1
                elif bullet_directions[i] == 'left':
                    dx, dy = -1, 0
                elif bullet_directions[i] == 'right':
                    dx, dy = 1, 0
                elif bullet_directions[i] == 'up_left':
                    dx, dy = -1, -1
                elif bullet_directions[i] == 'up_right':
                    dx, dy = 1, -1
                elif bullet_directions[i] == 'down_left':
                    dx, dy = -1, 1
                elif bullet_directions[i] == 'down_right':
                    dx, dy = 1, 1
                else:
                    return float('-inf')
                bullet_directions[i] = (dx, dy)

            avoidance_score = 0
            for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
                # Predict the next position of the bullet
                next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])

                # Check if the bullet is aimed at the new position
                if next_bullet_pos == new_pos:
                    return -500

                distance_to_bullet = chebyshev_distance(new_pos, bullet_pos)
                avoidance_score -= np.exp(-distance_to_bullet)  # Exponential penalty for proximity

            return np.exp(avoidance_score)

        # Ammo management score (higher reward for lower ammo count)=
        ammo_reward_factor = 1 - (state[2] / MAX_SHOTS)  # Linearly scale from 1 (min ammo) to 0 (max ammo)

        # calculate the new position after the move
        x, y = state[0], state[1]
        if action == 'MOVE_UP' or action == 'up':
            y -= 1
        elif action == 'MOVE_DOWN' or action == 'down':
            y += 1
        elif action == 'MOVE_LEFT' or action == 'left':
            x -= 1
        elif action == 'MOVE_RIGHT' or action == 'right':
            x += 1
        elif action == 'MOVE_UP_LEFT' or action == 'up_left':
            x -= 1
            y -= 1
        elif action == 'MOVE_UP_RIGHT' or action == 'up_right':
            x += 1
            y -= 1
        elif action == 'MOVE_DOWN_LEFT' or action == 'down_left':
            x -= 1
            y += 1
        elif action == 'MOVE_DOWN_RIGHT' or action == 'down_right':
            x += 1
            y += 1

        # Calculate proximity score to the opponent
        distance_to_opponent_old = chebyshev_distance((state[0], state[1]), (state[3], state[4]))
        distance_to_opponent_new = chebyshev_distance((x, y), (state[3], state[4]))

        # Reward for being at Chebyshev distance 3, penalize for less or more
        if distance_to_opponent_new < 2:
            return -1000
        else:
            proximity_score = (distance_to_opponent_old / distance_to_opponent_new) - 1

        # Bullet avoidance score
        avoidance_score = bullet_avoidance_score((x, y)) - bullet_avoidance_score((state[0], state[1]))

        # Combine scores with weighted factors
        avoidance_coef = 0.3 * len(self.board.bullets)
        other_coef = 0.3 * MAX_SHOTS - len(self.board.bullets)
        score = (ammo_reward_factor * (0.3 + other_coef)) + (proximity_score * (0.4 + other_coef)) + (
                avoidance_score * avoidance_coef)

        return score

    """
        factors:
        - bullets around the opponent
        - ammo for each tank
        - distance to the opponent
        - bullets aimed at you
        """

    def reward(self, new_state):
        """
        Get the reward for a given state-action pair.

        :param state: Current state.
        :param action: Action taken.
        :return: Reward for the state-action pair.
        """

        def count_bullets_in_area(opponent_pos):
            """
            Counts the number of bullets within a 3x3 area around the opponent.
            """

            bullet_positions = []
            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                i += 5

            x_min = max(opponent_pos[0] - 1, 0)
            x_max = min(opponent_pos[0] + 1, self.board.size - 1)
            y_min = max(opponent_pos[1] - 1, 0)
            y_max = min(opponent_pos[1] + 1, self.board.size - 1)

            bullets_in_area = 0

            for bullet_pos in bullet_positions:
                if x_min <= bullet_pos[0] <= x_max and y_min <= bullet_pos[1] <= y_max:
                    bullets_in_area += 1

            return bullets_in_area

        def chebyshev_distance(pos1, pos2):
            """
            Calculate the Chebyshev distance between two positions.
            """
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

        def bullet_avoidance_score(pos):
            """
            Calculate the bullet avoidance score based on bullets aimed at the new position
            and their distances.
            """
            bullet_positions = []
            bullet_directions = []

            i = 6
            while i < len(state):
                bullet_positions.append((state[i], state[i + 1]))
                bullet_directions.append(state[i + 2])
                i += 5

            for i in range(len(bullet_directions)):
                if bullet_directions[i] == 'up':
                    dx, dy = 0, -1
                elif bullet_directions[i] == 'down':
                    dx, dy = 0, 1
                elif bullet_directions[i] == 'left':
                    dx, dy = -1, 0
                elif bullet_directions[i] == 'right':
                    dx, dy = 1, 0
                elif bullet_directions[i] == 'up_left':
                    dx, dy = -1, -1
                elif bullet_directions[i] == 'up_right':
                    dx, dy = 1, -1
                elif bullet_directions[i] == 'down_left':
                    dx, dy = -1, 1
                elif bullet_directions[i] == 'down_right':
                    dx, dy = 1, 1
                else:
                    return float('-inf')
                bullet_directions[i] = (dx, dy)

            avoidance_score = 0
            for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
                for i in range(5):
                    distance_to_bullet = chebyshev_distance(new_pos, bullet_pos)
                    avoidance_score -= np.exp(-distance_to_bullet * i)  # Exponential penalty for proximity

                    bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])

            return np.exp(avoidance_score)

        # Unpack the state
        x, y, shots, opponent_x, opponent_y, opponent_shots = state

        bullet_around_opponent = count_bullets_in_area((opponent_x, opponent_y))

        bullets_aimed_at_you = bullet_avoidance_score((x, y))
        bullets_aimed_at_opponent = bullet_avoidance_score((opponent_x, opponent_y))

        # Reward factors
        ammo_reward_factor = 0.2
        proximity_reward_factor = 0.4
        avoidance_reward_factor = 0.4

        # Calculate the reward
        reward = (ammo_reward_factor * (shots - opponent_shots)) + (proximity_reward_factor * bullet_around_opponent) + (avoidance_reward_factor * (bullets_aimed_at_you - bullets_aimed_at_opponent))

        return reward

    def get_state(self):
        """
        Get the current state of the tank.

        :return: Tuple representing the state.
        """
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        state = [self.x, self.y, self.shots, target_tank.x, target_tank.y, target_tank.shots]
        for bullet in self.board.bullets:
            state.extend([bullet.x, bullet.y, bullet.direction, bullet.bounces, bullet.moves])
        return state

    def next_state(self, state, action):
        next_state = state.copy()
        if action == 'MOVE_UP':
            next_state[1] -= 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_DOWN':
            next_state[1] += 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_LEFT':
            next_state[0] -= 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_RIGHT':
            next_state[0] += 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_UP_LEFT':
            next_state[0] -= 1
            next_state[1] -= 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_UP_RIGHT':
            next_state[0] += 1
            next_state[1] -= 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_DOWN_LEFT':
            next_state[0] -= 1
            next_state[1] += 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'MOVE_DOWN_RIGHT':
            next_state[0] += 1
            next_state[1] += 1
            next_state[2] = min(MAX_SHOTS, next_state[2] + 1)
        elif action == 'SHOOT_UP':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0], next_state[1] - 1, 'up', 0, 0])
        elif action == 'SHOOT_DOWN':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0], next_state[1] + 1, 'down', 0, 0])
        elif action == 'SHOOT_LEFT':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] - 1, next_state[1], 'left', 0, 0])
        elif action == 'SHOOT_RIGHT':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] + 1, next_state[1], 'right', 0, 0])
        elif action == 'SHOOT_UP_LEFT':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] - 1, next_state[1] - 1, 'up_left', 0, 0])
        elif action == 'SHOOT_UP_RIGHT':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] + 1, next_state[1] - 1, 'up_right', 0, 0])
        elif action == 'SHOOT_DOWN_LEFT':
            next_state[2 + prefix] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] - 1, next_state[1] + 1, 'down_left', 0, 0])
        elif action == 'SHOOT_DOWN_RIGHT':
            next_state[2] = max(0, next_state[2] - 1)
            next_state.extend([next_state[0] + 1, next_state[1] + 1, 'down_right', 0, 0])
        i = 6
        while i < len(next_state):
            dx, dy = str_to_vals[next_state[i + 2]]
            next_state[i] += dx
            next_state[i + 1] += dy
            # Handle wall bounces
            if next_state[i] < 0 or next_state[i] >= self.board.size:
                dx = -dx  # Bounce off vertical walls
                next_state[i] += 2 * dx
                next_state[i + 2] = vals_to_str[-dx, dy]
                next_state[i + 3] += 1
                if next_state[i + 3] >= 3:
                    next_state = next_state[:i] + next_state[i + 5:]
                    continue
            if next_state[i + 1] < 0 or next_state[i + 1] >= self.board.size:
                dy = -dy  # Bounce off horizontal walls
                next_state[i + 1] += 2 * dy
                next_state[i + 2] = vals_to_str[dx, -dy]
                next_state[i + 3] += 1
                if next_state[i + 3] >= 3:
                    next_state = next_state[:i] + next_state[i + 5:]
                    continue
            next_state[i + 4] += 1
            if next_state[i + 4] >= 10:
                next_state = next_state[:i] + next_state[i + 5:]
            i += 5
        return next_state

    def minimax(self, state):
        """
        Minimax algorithm to determine the best move.

        :param state: State of the tank (x, y).
        :param depth: Depth of the search tree.
        :param maximizing_player: Whether the current player is maximizing or minimizing.
        :return: Best value for the current player.
        """

        def max_value(state, depth, alpha, beta):
            state[0], state[3] = state[3], state[0]
            state[1], state[4] = state[4], state[1]
            state[2], state[5] = state[5], state[2]
            if depth == 0 or state.done():
                return self.evaluate_game_state(state)
            v = float('-inf')
            for action in state.get_legal_actions(self.number):
                v = max(v, min_value(self.next_state(state, action), depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, alpha, beta):
            state[0], state[3] = state[3], state[0]
            state[1], state[4] = state[4], state[1]
            state[2], state[5] = state[5], state[2]
            if depth == 0 or state.done():
                return self.evaluate_game_state(state)
            v = float('inf')
            other_tank = 1 if self.number == 2 else 2
            for action in state.get_legal_actions(other_tank):
                v = min(v, max_value(self.next_state(state, action), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        scores = {}
        for action in self.get_legal_actions(self.number):
            state = [self.x, self.y, self.shots, self.board.tank2.x, self.board.tank2.y, self.board.tank2.shots]
            state[0], state[3] = state[3], state[0]
            state[1], state[4] = state[4], state[1]
            state[2], state[5] = state[5], state[2]
            for bullet in self.board.bullets:
                state.extend([bullet.x, bullet.y, bullet.direction, bullet.bounces, bullet.moves])
            score = min_value(state, self.depth, alpha, beta)
            # restore the data of each tank
            if score > best_score:
                best_score = score
                best_action = action
            scores[action] = score
        return best_action

    def move(self, action):
        """
        Move the tank using Minimax algorithm to reach the goal.
        """
        super(MinimaxTank, self).move(action)
        next_state = (-1, -1)
        if action == 'MOVE_UP':
            next_state = (self.x, self.y - 1)
        elif action == 'MOVE_DOWN':
            next_state = (self.x, self.y + 1)
        elif action == 'MOVE_LEFT':
            next_state = (self.x - 1, self.y)
        elif action == 'MOVE_RIGHT':
            next_state = (self.x + 1, self.y)
        elif action == 'MOVE_UP_LEFT':
            next_state = (self.x - 1, self.y - 1)
        elif action == 'MOVE_UP_RIGHT':
            next_state = (self.x + 1, self.y - 1)
        elif action == 'MOVE_DOWN_LEFT':
            next_state = (self.x - 1, self.y + 1)
        elif action == 'MOVE_DOWN_RIGHT':
            next_state = (self.x + 1, self.y + 1)
        moved = self.board.move_tank(self, next_state[0], next_state[1], self.number)
        if moved:
            self.x, self.y = next_state
            return True
        return False

    def shoot(self, action):
        """
        Shoot a bullet at the target tank if within range.
        """
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        tank_position = (target_tank.x, target_tank.y)
        if abs(self.x - tank_position[0]) <= 1 and abs(self.y - tank_position[1]) <= 1:
            if action == 'SHOOT_UP':
                dx, dy = 0, -1
            elif action == 'SHOOT_DOWN':
                dx, dy = 0, 1
            elif action == 'SHOOT_LEFT':
                dx, dy = -1, 0
            elif action == 'SHOOT_RIGHT':
                dx, dy = 1, 0
            elif action == 'SHOOT_UP_LEFT':
                dx, dy = -1, -1
            elif action == 'SHOOT_UP_RIGHT':
                dx, dy = 1, -1
            elif action == 'SHOOT_DOWN_LEFT':
                dx, dy = -1, 1
            elif action == 'SHOOT_DOWN_RIGHT':
                dx, dy = 1, 1
            else:
                return False
            direction_map = {
                (0, -1): 'up',
                (0, 1): 'down',
                (-1, 0): 'left',
                (1, 0): 'right',
                (-1, -1): 'up_left',
                (1, -1): 'up_right',
                (-1, 1): 'down_left',
                (1, 1): 'down_right'
            }
            direction = direction_map.get((dx, dy))
            if direction:
                self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, direction))
                return True
        return False

    def update(self):
        if self.number == 1:
            game_state = self.get_state()
        else:
            game_state = self.get_state()
        action = self.minimax(game_state)

        if action.startswith('MOVE'):
            self.move(action)
        else:
            self.shoot(action)
        print(self.x, self.y, self.shots, action)
        # for i in range(10):
        #     for j in range(10):
        #         print(self.board.grid[i][j], end=' ')
        #     print()
