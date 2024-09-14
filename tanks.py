from abc import ABC, abstractmethod
from bullet import Bullet
import numpy as np
import random
from game_constants import GameConstants
import heapq
import pickle

from planning_problem import PlanningProblem, solve, null_heuristic, max_level, level_sum


# ---------------------- Utilities ---------------------- #


def chebyshev_distance(pos1, pos2):
    """
    Calculate the Chebyshev distance between two positions.
    """
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def a_star_path(board, start, goal):
    """
    Compute the A* path from start to goal.

    :param start: Starting position (x, y).
    :param goal: Goal position (x, y).
    :return: List of positions (x, y) in the path.
    """

    def heuristic(a, b):
        # Chebyshev distance
        return chebyshev_distance(a, b)

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

        for dx, dy in GameConstants.VALS_TO_STR.keys():
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < board.width and 0 <= neighbor[1] < board.height and \
                    not board.is_wall(neighbor[0], neighbor[1]) \
                    and not any([bullet.x == neighbor[0] and bullet.y == neighbor[1] for bullet in
                                 board.bullets]):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []


def state_legal_actions(board, state, prefix):
    """
    Get the legal actions for a given state.

    :param state: Current state.
    :return: List of legal actions.
    """
    legal_actions = []
    for action in GameConstants.ACTIONS:
        # making sure the tank is not moving or shooting into a wall
        if action.startswith('MOVE'):
            simp_action = action[5:].lower()
        else:
            simp_action = action[6:].lower()
        dx, dy = GameConstants.STR_TO_VALS[simp_action]
        if state[0 + prefix] + dx < 0 or state[0 + prefix] + dx >= board.width or \
                state[1 + prefix] + dy < 0 or state[1 + prefix] + dy >= board.height or \
                board.is_wall(state[0 + prefix] + dx, state[1 + prefix] + dy):
            continue
        # making sure the tank is not moving into a bullet or another tank
        if action.startswith('MOVE'):
            illegal = False
            for i in range(6, len(state), 5):
                if state[i] == state[0 + prefix] + dx and state[i + 1] == state[1 + prefix] + dy:
                    illegal = True
                    break
            if illegal:
                continue
            if state[3 - prefix] == (state[0 + prefix] + dx) and state[4 - prefix] == (state[1 + prefix] + dy):
                continue
        else:
            # making sure the bullet's starting position is legal
            x, y = state[0 + prefix], state[1 + prefix]
            x += dx
            y += dy
            if not (board.grid[y][x] == GameConstants.BOARD or
                    board.grid[y][x] == GameConstants.TANK1 or
                    board.grid[y][x] == GameConstants.TANK2):
                continue

        # checking specific move legality
        if action == 'MOVE_UP' or (action == 'SHOOT_UP' and state[2 + prefix] > 0):
            if 0 < state[1 + prefix] < board.height:
                legal_actions.append(action)
        elif action == 'MOVE_DOWN' or (action == 'SHOOT_DOWN' and state[2 + prefix] > 0):
            if 0 <= state[1 + prefix] < board.height - 1:
                legal_actions.append(action)
        elif action == 'MOVE_LEFT' or (action == 'SHOOT_LEFT' and state[2 + prefix] > 0):
            if 0 < state[0 + prefix] < board.width:
                legal_actions.append(action)
        elif action == 'MOVE_RIGHT' or (action == 'SHOOT_RIGHT' and state[2 + prefix] > 0):
            if 0 <= state[0 + prefix] < board.width - 1:
                legal_actions.append(action)
        elif action == 'MOVE_UP_LEFT' or (action == 'SHOOT_UP_LEFT' and state[2 + prefix] > 0):
            if 0 < state[0 + prefix] < board.width and 0 < state[1 + prefix] < board.height:
                legal_actions.append(action)
        elif action == 'MOVE_UP_RIGHT' or (action == 'SHOOT_UP_RIGHT' and state[2 + prefix] > 0):
            if 0 <= state[0 + prefix] < board.width - 1 and 0 < state[1 + prefix] < board.height:
                legal_actions.append(action)
        elif action == 'MOVE_DOWN_LEFT' or (action == 'SHOOT_DOWN_LEFT' and state[2 + prefix] > 0):
            if 0 < state[0 + prefix] < board.width - 1 and 0 <= state[1 + prefix] < board.height - 1:
                legal_actions.append(action)
        elif action == 'MOVE_DOWN_RIGHT' or (action == 'SHOOT_DOWN_RIGHT' and state[2 + prefix] > 0):
            if 0 <= state[0 + prefix] < board.width - 1 and 0 <= state[1 + prefix] < board.height - 1:
                legal_actions.append(action)
    return legal_actions


def clear_shot(board, pos1, pos2):
    """
    Check if there is a clear shot between two positions.
    """
    dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
    if dx == 0:
        for i in range(1, abs(dy)):
            if board.is_wall(pos1[0], pos1[1] + i * dy // abs(dy)):
                return False
    elif dy == 0:
        for i in range(1, abs(dx)):
            if board.is_wall(pos1[0] + i * dx // abs(dx), pos1[1]):
                return False
    else:
        if dx != dy:
            return False
        for i in range(1, abs(dx)):
            if board.is_wall(pos1[0] + i * dx // abs(dx), pos1[1] + i * dy // abs(dy)):
                return False
    return True


def get_bullet_info(state):
    """
    Get the bullet information from the state.
    """
    bullet_positions = []
    bullet_directions = []

    i = 6
    while i < len(state):
        bullet_positions.append((state[i], state[i + 1]))
        direction = state[i + 2]
        if direction.startswith('SHOOT'):
            direction = direction[6:]
            direction = direction.lower()
        bullet_directions.append(direction)
        i += 5

    for i in range(len(bullet_directions)):
        dx, dy = GameConstants.STR_TO_VALS[bullet_directions[i]]
        bullet_directions[i] = (dx, dy)

    return bullet_positions, bullet_directions


def bullet_avoidance_score(state, new_pos):
    """
    Calculate the bullet avoidance score based on bullets aimed at the new position
    and their distances.
    """
    bullet_positions, bullet_directions = get_bullet_info(state)

    avoidance_score = 0
    for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
        # Predict the next position of the bullet
        next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])

        # Check if the bullet is aimed at the new position
        if next_bullet_pos == new_pos:
            return -500

        distance_to_bullet = chebyshev_distance(new_pos, bullet_pos)
        avoidance_score -= 100 * np.exp(-distance_to_bullet)  # Exponential penalty for proximity

    return np.exp(avoidance_score)


def next_state_finder(board, state, action, prefix):
    """
    Get the next state after taking an action.

    :param state: Current state.
    :param action: Action taken.
    :param prefix: Prefix for the state, which means which tank is being moved.
    :return: Next state.
    """
    next_state = state.copy()

    # update bullets
    i = 6
    while i < len(next_state):
        simp_action = next_state[i + 2]
        if simp_action.startswith('SHOOT'):
            simp_action = simp_action[6:]
            simp_action = simp_action.lower()
        dx, dy = GameConstants.STR_TO_VALS[simp_action]
        next_state[i] += dx
        next_state[i + 1] += dy
        # Handle wall bounces
        if next_state[i] < 0 or next_state[i] >= board.width:
            dx = -dx  # Bounce off vertical walls
            next_state[i] += 2 * dx
            next_state[i + 2] = GameConstants.VALS_TO_STR[-dx, dy]
            next_state[i + 3] += 1
            if next_state[i + 3] >= 3:
                next_state = next_state[:i] + next_state[i + 5:]
                continue
        if next_state[i + 1] < 0 or next_state[i + 1] >= board.height:
            dy = -dy  # Bounce off horizontal walls
            next_state[i + 1] += 2 * dy
            next_state[i + 2] = GameConstants.VALS_TO_STR[dx, -dy]
            next_state[i + 3] += 1
            if next_state[i + 3] >= 3:
                next_state = next_state[:i] + next_state[i + 5:]
                continue
        next_state[i + 4] += 1
        if next_state[i + 4] >= 10:
            next_state = next_state[:i] + next_state[i + 5:]
        i += 5

    # update the tank based on the action
    if action == 'MOVE_UP':
        next_state[1 + prefix] -= 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_DOWN':
        next_state[1 + prefix] += 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_LEFT':
        next_state[0 + prefix] -= 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_RIGHT':
        next_state[0 + prefix] += 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_UP_LEFT':
        next_state[0 + prefix] -= 1
        next_state[1 + prefix] -= 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_UP_RIGHT':
        next_state[0 + prefix] += 1
        next_state[1 + prefix] -= 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_DOWN_LEFT':
        next_state[0 + prefix] -= 1
        next_state[1 + prefix] += 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'MOVE_DOWN_RIGHT':
        next_state[0 + prefix] += 1
        next_state[1 + prefix] += 1
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)
    elif action == 'SHOOT_UP':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix], next_state[1 + prefix] - 1, 'up', 0, 0])
    elif action == 'SHOOT_DOWN':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix], next_state[1 + prefix] + 1, 'down', 0, 0])
    elif action == 'SHOOT_LEFT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] - 1, next_state[1 + prefix], 'left', 0, 0])
    elif action == 'SHOOT_RIGHT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] + 1, next_state[1 + prefix], 'right', 0, 0])
    elif action == 'SHOOT_UP_LEFT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] - 1, next_state[1 + prefix] - 1, 'up_left', 0, 0])
    elif action == 'SHOOT_UP_RIGHT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] + 1, next_state[1 + prefix] - 1, 'up_right', 0, 0])
    elif action == 'SHOOT_DOWN_LEFT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] - 1, next_state[1 + prefix] + 1, 'down_left', 0, 0])
    elif action == 'SHOOT_DOWN_RIGHT':
        next_state[2 + prefix] = max(0, next_state[2 + prefix] - 1)
        next_state.extend([next_state[0 + prefix] + 1, next_state[1 + prefix] + 1, 'down_right', 0, 0])
    elif action == 'IDLE':
        next_state[2 + prefix] = min(GameConstants.MAX_SHOTS, next_state[2 + prefix] + 1)

    return next_state


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
        self.shots = GameConstants.BEGINNING_SHOTS  # Shot counter
        self.number = number  # Tank number
        self.bullets = []  # List of bullets
        board.place_tank(self, number)

    def add_bullet(self):
        """ Add a bullet to the tank's shot counter. """
        self.shots += 1 if self.shots < GameConstants.MAX_SHOTS else 0

    @abstractmethod
    def move(self, direction):
        """Move the tank in a specified direction."""
        self.add_bullet()  # add one bullet each time the tank moves
        pass

    @abstractmethod
    def shoot(self, direction):
        """Shoot a bullet in a specified direction."""
        pass

    @abstractmethod
    def act(self):
        """Perform an action."""
        pass

    def __get_legal_actions_shoot(self):
        legal_actions = []
        if self.y > 0:
            legal_actions.append('SHOOT_UP')
        if self.y < self.board.height - 1:
            legal_actions.append('SHOOT_DOWN')
        if self.x > 0:
            legal_actions.append('SHOOT_LEFT')
        if self.x < self.board.width - 1:
            legal_actions.append('SHOOT_RIGHT')
        if self.y > 0 and self.x > 0:
            legal_actions.append('SHOOT_UP_LEFT')
        if self.y > 0 and self.x < self.board.width - 1:
            legal_actions.append('SHOOT_UP_RIGHT')
        if self.y < self.board.height - 1 and self.x > 0:
            legal_actions.append('SHOOT_DOWN_LEFT')
        if self.y < self.board.height - 1 and self.x < self.board.width - 1:
            legal_actions.append('SHOOT_DOWN_RIGHT')
        return legal_actions

    def __get_legal_actions_move(self):
        legal_actions = []
        if self.y > 0:
            legal_actions.append('MOVE_UP')
        if self.y < self.board.height - 1:
            legal_actions.append('MOVE_DOWN')
        if self.x > 0:
            legal_actions.append('MOVE_LEFT')
        if self.x < self.board.width - 1:
            legal_actions.append('MOVE_RIGHT')
        if self.y > 0 and self.x > 0:
            legal_actions.append('MOVE_UP_LEFT')
        if self.y > 0 and self.x < self.board.width - 1:
            legal_actions.append('MOVE_UP_RIGHT')
        if self.y < self.board.height - 1 and self.x > 0:
            legal_actions.append('MOVE_DOWN_LEFT')
        if self.y < self.board.height - 1 and self.x < self.board.width - 1:
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


class AdversarialSearchTank(Tank, ABC):

    def __init__(self, board, x, y, number):
        """
        Initialize an AI-controlled tank using the Minimax algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)
        self.depth = 2

    def evaluate_game_state(self, state):
        """
        Get the reward for a given state-action pair.

        :param state: Current state.
        :param action: Action taken.
        :return: Reward for the state-action pair.
        """

        # Unpack the state
        x, y, shots, opponent_x, opponent_y, opponent_shots = state[0], state[1], state[2], state[3], state[4], state[5]

        # check if there is a bullet in the current position
        i = 6
        while i < len(state):
            if state[i] == x and state[i + 1] == y:
                return -1000
            if state[i] == opponent_x and state[i + 1] == opponent_y:
                return 1000
            i += 5

        # Calculate the A* distance to the opponent
        path = len(a_star_path(self.board, (x, y), (opponent_x, opponent_y)))
        if path <= 1:
            distance_score = -100
        elif path > 1:
            distance_score = 100 / path

        # checking if there is a clean shot
        if clear_shot(self.board, (x, y), (opponent_x, opponent_y)) and state[2] > 0:
            clear_shot_score = 50
            path = a_star_path(self.board, (x, y), (opponent_x, opponent_y))
            for i in range(6, len(state), 5):
                if state[i] == x and state[i + 1] == y:
                    clear_shot_score = 100
        else:
            clear_shot_score = 0

        # bullet management
        if shots > 0:
            bullet_management = 5 * state[2]
        else:
            bullet_management = -10

        # bullet avoidance
        bullet_avoidance = bullet_avoidance_score(state, (x, y))

        return distance_score + clear_shot_score + bullet_management + bullet_avoidance

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

    def is_terminal_state(self, state):
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

    def search(self, state):
        """
        Minimax/Expectimax algorithm to determine the best move.

        :param state: State of the tank.
        :return: Best value for the current player.
        """

        raise NotImplementedError

    def move(self, action):
        """
        Move the tank using Minimax algorithm to reach the goal.
        :param action: Action to take.
        """
        super(AdversarialSearchTank, self).move(action)
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
        :param action: Action to take.
        """
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        tank_position = (target_tank.x, target_tank.y)
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
        direction = GameConstants.VALS_TO_STR.get((dx, dy))
        if direction:
            self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, direction))
            return True
        return False

    def act(self):
        game_state = self.get_state()
        action = self.search(game_state)

        if action.startswith('MOVE'):
            self.move(action)
        else:
            self.shoot(action)


# ---------------------- RandomTank ---------------------- #
class RandomTank(Tank):
    def __init__(self, board, x, y, number):
        """
        Initialize a tank with random legal moves.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)

    def move(self, _):
        """
        Move the tank in a random direction.
        :param _: Unused parameter. Needed to match the parent class signature.
        :return: True if move is valid, False otherwise.
        """
        super().move(_)
        legal_actions = self.get_legal_actions()
        valid_actions = [action for action in legal_actions if action.startswith('MOVE')]
        while valid_actions:
            action = np.random.choice(valid_actions)
            dx, dy = GameConstants.STR_TO_VALS[action[5:].lower()]
            if self.board.is_valid_move(self.x + dx, self.y + dy) and not self.board.is_tank(self.x + dx, self.y + dy):
                self.board.move_tank(self, self.x + dx, self.y + dy, self.number)
                return True
            valid_actions.remove(action)
        return False

    def shoot(self, _):
        """
        Shoot a bullet in a random direction.
        :param _: Unused parameter. Needed to match the parent class signature.
        :return: True if shoot is valid, False otherwise.
        """
        super().shoot(_)
        legal_actions = self.get_legal_actions()
        valid_actions = [action for action in legal_actions if action.startswith('SHOOT')]
        while valid_actions:
            action = np.random.choice(valid_actions)
            direction = action[6:].lower()
            dx, dy = GameConstants.STR_TO_VALS[direction]
            bullet = Bullet(self.board, self.x + dx, self.y + dy, direction)
            can_add = self.board.is_valid_move(bullet.x, bullet.y)
            if can_add:
                self.board.add_bullet(bullet)
                self.shots -= 1
                return True
        return False

    def act(self):
        # choose randomly between moving and shooting - below 0.8 move, above 0.8 shoot
        import random
        choice = random.random()
        if choice < 0.8:
            self.move(None)
        else:
            self.shoot(None)


# ---------------------- PlayerTank ---------------------- #
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
        if direction in GameConstants.STR_TO_VALS:
            dx, dy = GameConstants.STR_TO_VALS[direction]
            new_x, new_y = self.x + dx, self.y + dy
            return self.board.move_tank(self, new_x, new_y, self.number)
        return False

    def shoot(self, direction):
        """
        Shoot a bullet in a specified direction.

        :param direction: Direction to shoot ('up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right').
        """
        if self.shots > 0:
            if direction in GameConstants.STR_TO_VALS:
                dx, dy = GameConstants.STR_TO_VALS[direction]
                bullet = Bullet(self.board, self.x + dx, self.y + dy, direction)
                can_add = self.board.add_bullet(bullet)
                if can_add:
                    self.shots -= 1
                return can_add
        else:
            self.board.show_message("You can't shoot yet!")
            return False

    def act(self):
        pass


# ---------------------- A*Tank ---------------------- #
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
        path = a_star_path(self.board, (self.x, self.y), tank_position)
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
            dx, dy = tank_position[0] - self.x, tank_position[1] - self.y
            direction = GameConstants.VALS_TO_STR.get((dx, dy))
            if direction:
                self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, direction))
                self.shots -= 1

    def legal(self, action):
        if action.startswith('MOVE'):
            action = action[5:]
        else:
            action = action[6:]
        action = action.lower()
        dx, dy = GameConstants.STR_TO_VALS[action]
        new_x, new_y = self.x + dx, self.y + dy
        return self.board.is_valid_move(new_x, new_y)

    def act(self):
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        if chebyshev_distance((self.x, self.y), (target_tank.x, target_tank.y)) == 1:
            self.shoot(None)
            return
        moved = self.move(None)
        if not moved:  # if there is no path to the target tank
            import random
            action = random.choice(GameConstants.ACTIONS)
            while not self.legal(action):
                action = random.choice(GameConstants.ACTIONS)
            if action.startswith('MOVE'):
                action = action[5:]
                action = action.lower()
                dx, dy = GameConstants.STR_TO_VALS[action]
                self.board.move_tank(self, self.x + dx, self.y + dy, self.number)
            else:
                action = action[6:]
                action = action.lower()
                dx, dy = GameConstants.STR_TO_VALS[action]
                self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, action))
                self.shots -= 1


# ---------------------- Q-LearningTank ---------------------- #
class QLearningTank(Tank):
    def __init__(self, board, x, y, number, lr=0.2, ds=0.99, er=0.1, ed=0.01, epochs=100, pretrained=False,
                 save_file=None):
        """
        Initialize an AI-controlled tank using the Q-learning algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        :param lr: Learning rate.
        :param ds: Discount factor.
        :param er: Exploration rate.
        :param ed: Exploration decay.
        :param epochs: Number of epochs.
        :param pretrained: Load pretrained model.
        :param save_file: File to save the model.
        """
        super().__init__(board, x, y, number)
        self.q_table = {}
        self.learning_rate = lr
        self.discount_factor = ds
        self.exploration_rate = er
        self.exploration_decay = ed
        self.epochs = epochs
        self.save_file = save_file
        self.pretrained = pretrained

        if self.pretrained:
            # load from pickle file
            with open(self.save_file, 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.fill_q_table()

    def fill_q_table(self):
        """
        Fill the Q-table with the given state.
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

        if self.pretrained:
            return

        for i in range(self.epochs):
            state = self.get_random_state()
            while not is_terminal_state(state):
                legal_actions = state_legal_actions(self.board, state, 0)
                if np.random.rand() < self.exploration_rate:
                    action = np.random.choice(legal_actions)
                else:
                    q_values = [self.get_q_value(tuple(state), action) for action in legal_actions]
                    if len(legal_actions) == 0:
                        action = "IDLE"
                    else:
                        action = legal_actions[np.argmax(q_values)]
                reward = self.reward(state, action)
                next_state = self.next_state_wrapper(list(state), action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.exploration_rate *= (1 - self.exploration_decay)

        if self.save_file != None:
            # save the Q-table to a pickle file
            with open(self.save_file, 'wb') as f:
                pickle.dump(self.q_table, f)

    def illegal_state(self, tank1, tank2):
        """
        Check if the given state is illegal.

        :param tank1: Tank 1 position.
        :param tank2: Tank 2 position.
        :return: True if illegal, False otherwise.
        """
        if tank1[0] == tank2[0] and tank1[1] == tank2[1]:
            return True
        if self.board.is_wall(tank1[0], tank1[1]) or self.board.is_wall(tank2[0], tank2[1]):
            return True
        return False

    def get_random_state(self):
        """
        Get a random state for the Q-table.

        :return: Random state.
        """
        tank1 = (random.randint(0, self.board.width - 1), random.randint(0, self.board.height - 1))
        tank2 = (random.randint(0, self.board.width - 1), random.randint(0, self.board.height - 1))
        shots1 = random.randint(0, GameConstants.MAX_SHOTS)
        shots2 = random.randint(0, GameConstants.MAX_SHOTS)
        while self.illegal_state(tank1, tank2):
            tank1 = (random.randint(0, self.board.width - 1), random.randint(0, self.board.height - 1))
            tank2 = (random.randint(0, self.board.width - 1), random.randint(0, self.board.height - 1))
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

    def next_state_wrapper(self, state, action):
        """
        Get the next state after taking an action and the opponent's move.

        :param state: Current state.
        :param action: Action taken.
        :return: Next state.
        """
        next_state = next_state_finder(self.board, state, action, 0)
        next_state = next_state_finder(self.board, next_state, self.state_to_move(next_state), 3)
        return tuple(next_state)

    def state_to_move(self, state):
        """
        Get the move corresponding to a state.

        :param state: State to convert.
        :return: Move corresponding to the state.
        """
        q_values = [self.get_q_value(tuple(state), action) for action in state_legal_actions(self.board, state, 3)]
        if len(q_values) == 0:
            return "IDLE"
        action = state_legal_actions(self.board, state, 3)[np.argmax(q_values)]
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
        q_values = [self.get_q_value(tuple(state), action) for action in state_legal_actions(self.board, state, 0)]
        if len(q_values) == 0:
            return "IDLE"
        action = state_legal_actions(self.board, state, 0)[np.argmax(q_values)]
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
        next_q_values = [self.get_q_value(next_state, next_action) for next_action in GameConstants.ACTIONS]
        max_q_value = np.max(next_q_values)
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_q_value - q_value)
        self.q_table[(state, action)] = new_q_value

    def calculate_shoot_reward(self, state, action):

        if state[2] == 0:
            return float('-inf')  # illegal move

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

        # Checking for an immediate hit
        path = len(a_star_path(self.board, (state[0], state[1]), (state[3], state[4])))
        if path == 1:
            if chebyshev_distance((state[0] + dx, state[1] + dy), (state[3], state[4])) == 0:
                return 1000
            return -1000

        if clear_shot(self.board, (state[0], state[1]), (state[3], state[4])):
            shot_score = (14 - len(a_star_path(self.board, (state[0], state[1]), (state[3], state[4])))) * 10
        else:
            shot_score = -20

        # check if there is a bullet aimed at the tank
        bullet_positions, bullet_directions = get_bullet_info(state)

        avoidance_score = 0
        for bullet_pos, bullet_dir in zip(bullet_positions, bullet_directions):
            # it is checked twice because the bullet moves in both players' turn in a turn

            # Predict the next position of the bullet
            next_bullet_pos = (bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1])
            if next_bullet_pos == (state[0], state[1]):
                return -1000

            # handle bullet bounces
            if next_bullet_pos[0] < 0 or next_bullet_pos[0] >= self.board.width:
                bullet_dir = (-bullet_dir[0], bullet_dir[1])
            if next_bullet_pos[1] < 0 or next_bullet_pos[1] >= self.board.height:
                bullet_dir = (bullet_dir[0], -bullet_dir[1])

            next_bullet_pos = (next_bullet_pos[0] + bullet_dir[0], next_bullet_pos[1] + bullet_dir[1])
            # Check if the bullet is aimed at the new position
            if next_bullet_pos == (state[0], state[1]):
                return -1000

        return shot_score

    def calculate_move_reward(self, state, action):

        # Ammo management score (higher reward for lower ammo count)=
        ammo_reward_factor = 1 - (
                state[2] / GameConstants.MAX_SHOTS)  # Linearly scale from 1 (min ammo) to 0 (max ammo)

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
        distance_to_opponent_old = len(a_star_path(self.board, (state[0], state[1]), (state[3], state[4])))
        distance_to_opponent_new = len(a_star_path(self.board, (x, y), (state[3], state[4])))

        # Reward for being at Chebyshev distance of at least 3, penalize for less
        if distance_to_opponent_new < 2:
            return -1000
        else:
            proximity_score = 1 * (distance_to_opponent_old / distance_to_opponent_new) - 1

        # Bullet avoidance score
        avoidance_score = bullet_avoidance_score(state, (x, y)) - bullet_avoidance_score(state, (state[0], state[1]))

        # Combine scores with weighted factors
        avoidance_coef = 0.3 * len(self.board.bullets)
        other_coef = 0.3 * GameConstants.MAX_SHOTS - len(self.board.bullets)
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
        elif action.startswith('MOVE'):
            return self.calculate_move_reward(state, action)
        return 0  # only if there are no legal moves

    def move(self, action):
        """
        Move the tank using the Q-learning algorithm to reach the goal.

        :return: True if move is valid, False otherwise.
        """
        super(QLearningTank, self).move(action)

        state = self.get_state()
        next_state = self.next_state_wrapper(state, action)
        self.update_q_table(self.get_state(), action, self.reward(self.get_state(), action), next_state)
        return self.board.move_tank(self, next_state[0], next_state[1], self.number)

    def shoot(self, action):
        """
        Shoot a bullet at the target tank if within range.

        :param action: Action to take.
        :return: True if the shot is valid, False otherwise.
        """
        super(QLearningTank, self).shoot(action)

        simp_action = action[6:]
        simp_action = simp_action.lower()
        dx, dy = GameConstants.STR_TO_VALS[simp_action]

        state = self.get_state()
        next_state = self.next_state_wrapper(state, action)

        bullet = Bullet(self.board, self.x + dx, self.y + dy, simp_action)
        can_add = self.board.add_bullet(bullet)
        self.bullets.append(bullet)
        if can_add:
            self.shots -= 1
            self.update_q_table(self.get_state(), action, self.reward(self.get_state(), action), next_state)
        return can_add

    def act(self):
        """
        Update the tank's state.
        """
        action = self.choose_action(self.get_state())
        if action.startswith('MOVE'):
            self.move(action)
        elif action.startswith('SHOOT'):
            self.shoot(action)


# --------------------- MinimaxTank --------------------- #
class MinimaxTank(AdversarialSearchTank):
    def search(self, state):
        """
        Minimax algorithm to determine the best move.

        :param state: State of the tank (x, y).
        :param depth: Depth of the search tree.
        :param maximizing_player: Whether the current player is maximizing or minimizing.
        :return: Best value for the current player.
        """

        def max_value(state, depth, alpha, beta):
            if depth == 0 or self.is_terminal_state(state):
                return self.evaluate_game_state(state)
            v = float('-inf')
            actions = state_legal_actions(self.board, state, 0)
            for action in actions:
                v = max(v, min_value(next_state_finder(self.board, state, action, 0), depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, alpha, beta):
            if depth == 0 or self.is_terminal_state(state):
                return self.evaluate_game_state(state)
            v = float('inf')
            actions = state_legal_actions(self.board, state, 3)
            for action in actions:
                v = min(v, max_value(next_state_finder(self.board, state, action, 3), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        scores = {}
        for action in state_legal_actions(self.board, state, 0):
            state = self.get_state()
            state = next_state_finder(self.board, state, action, 0)
            score = min_value(state, self.depth, alpha, beta)
            # restore the data of each tank
            if score > best_score:
                best_score = score
                best_action = action
            scores[action] = score
        # print(scores)
        return best_action


# --------------------- ExpectimaxTank --------------------- #
class ExpectimaxTank(AdversarialSearchTank):
    def search(self, state):
        """
        Minimax algorithm to determine the best move.

        :param state: State of the tank (x, y).
        :param depth: Depth of the search tree.
        :param maximizing_player: Whether the current player is maximizing or minimizing.
        :return: Best value for the current player.
        """

        def max_value(state, depth):
            if depth == 0 or self.is_terminal_state(state):
                return self.evaluate_game_state(state)
            v = float('-inf')
            actions = state_legal_actions(self.board, state, 0)
            for action in actions:
                v = max(v, exp_value(next_state_finder(self.board, state, action, 0), depth))
            return v

        def exp_value(state, depth):
            if depth == 0 or self.is_terminal_state(state):
                return self.evaluate_game_state(state)
            v = 0
            actions = state_legal_actions(self.board, state, 3)
            for action in actions:
                v += max_value(next_state_finder(self.board, state, action, 3), depth - 1)
            return v / len(actions)

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        scores = {}
        for action in state_legal_actions(self.board, state, 0):
            state = self.get_state()
            state = next_state_finder(self.board, state, action, 0)
            score = exp_value(state, self.depth)
            # restore the data of each tank
            if score > best_score:
                best_score = score
                best_action = action
            scores[action] = score
        # print(scores)
        return best_action


# --------------------- Planning-GraphTank --------------------- #

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
        self.num_turns_to_replan = 1  # how many turns to wait before replanning,
        # for best - do 1, for faster - do more than 1
        self.decreasing_num_turns_to_replan = False  # just a small optimization

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

        if plan is None:
            self.isplan_uptodate = False
        else:
            self.plan = plan
            self.current_plan_index = 0
            self.isplan_uptodate = True
            # remove the _from_{x}_{y} suffix from every action in the plan

            self.plan = [action.name.split("_from")[0] for action in self.plan]
            # print(self.plan)
        if self.plan is None:  # in case of no plan found and also no previous plan
            self.plan = GameConstants.ACTIONS

    def create_domain_file(self, domain_file_name, board):
        # Create the domain file for the planning graph
        domain_file = open(domain_file_name, 'w')
        domain_file.write("Propositions:\n")
        # propositions: tank_at_x_y, enemy_at_x_y, bullet_at_x_y, wall_at_x_y, empty_at_x_y
        domain_file.write(f"null ")
        for x in range(0, board.width):
            for y in range(0, board.height):
                domain_file.write(f"tank_at_{x}_{y} ")
                # domain_file.write(f"enemy_at_{x}_{y} ")
                domain_file.write(f"bullet_at_{x}_{y} ")
                domain_file.write(f"wall_at_{x}_{y} ")
                domain_file.write(f"empty_at_{x}_{y} ")
        domain_file.write("\nActions:\n")
        # actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP_LEFT, MOVE_UP_RIGHT, MOVE_DOWN_LEFT, MOVE_DOWN_RIGHT
        #          SHOOT_UP, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT, SHOOT_UP_LEFT, SHOOT_UP_RIGHT, SHOOT_DOWN_LEFT, SHOOT_DOWN_RIGHT

        for x in range(board.width):
            for y in range(board.height):
                # if there is a wall at x,y then we can't move there
                if board.is_wall(x, y):
                    continue
                legal_actions = GameConstants.ACTIONS
                for action in legal_actions:
                    if "MOVE" in action:
                        if "UP_LEFT" in action:
                            if x == 0 or y == 0:
                                continue
                            elif board.is_wall(x - 1, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y - 1}\n")
                            domain_file.write(f"add: tank_at_{x - 1}_{y - 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x-1}_{y-1}\n")
                        elif "UP_RIGHT" in action:
                            if x == board.width - 1 or y == 0:
                                continue
                            elif board.is_wall(x + 1, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y - 1}\n")
                            domain_file.write(f"add: tank_at_{x + 1}_{y - 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x+1}_{y-1}\n")
                        elif "DOWN_LEFT" in action:
                            if x == 0 or y == board.height - 1:
                                continue
                            elif board.is_wall(x - 1, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y + 1}\n")
                            domain_file.write(f"add: tank_at_{x - 1}_{y + 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x-1}_{y+1}\n")
                        elif "DOWN_RIGHT" in action:
                            if x == board.width - 1 or y == board.height - 1:
                                continue
                            elif board.is_wall(x + 1, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y + 1}\n")
                            domain_file.write(f"add: tank_at_{x + 1}_{y + 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x+1}_{y+1}
                        elif "UP" in action:  # ~~~~ NOTE: the 4 direction must be after the diagonal directions
                            if y == 0:
                                continue
                            elif board.is_wall(x, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y - 1}\n")
                            domain_file.write(f"add: tank_at_{x}_{y - 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x}_{y-1}
                        elif "DOWN" in action:
                            if y == board.height - 1:
                                continue
                            elif board.is_wall(x, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y + 1}\n")
                            domain_file.write(f"add: tank_at_{x}_{y + 1}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x}_{y+1}\n")
                        elif "LEFT" in action:
                            if x == 0:
                                continue
                            elif board.is_wall(x - 1, y):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y}\n")
                            domain_file.write(f"add: tank_at_{x - 1}_{y}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x-1}_{y}\n")
                        elif "RIGHT" in action:
                            if x == board.width - 1:
                                continue
                            elif board.is_wall(x + 1, y):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y}\n")
                            domain_file.write(f"add: tank_at_{x + 1}_{y}\n")  # empty_at_{x}_{y}\n")
                            domain_file.write(f"delete: tank_at_{x}_{y}\n")  # empty_at_{x+1}_{y}\n")
        for x in range(board.width):
            for y in range(board.height):
                if board.is_wall(x, y):
                    continue
                legal_actions = GameConstants.ACTIONS
                for action in legal_actions:
                    # """

                    if "SHOOT" in action:
                        if "UP_LEFT" in action:
                            if x == 0 or y == 0:
                                continue
                            elif board.is_wall(x - 1, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y - 1}\n")
                            domain_file.write(f"add: bullet_at_{x - 1}_{y - 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "UP_RIGHT" in action:
                            if x == board.width - 1 or y == 0:
                                continue
                            elif board.is_wall(x + 1, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y - 1}\n")
                            domain_file.write(f"add: bullet_at_{x + 1}_{y - 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "DOWN_LEFT" in action:
                            if x == 0 or y == board.height - 1:
                                continue
                            elif board.is_wall(x - 1, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y + 1}\n")
                            domain_file.write(f"add: bullet_at_{x - 1}_{y + 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "DOWN_RIGHT" in action:
                            if x == board.width - 1 or y == board.height - 1:
                                continue
                            elif board.is_wall(x + 1, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y + 1}\n")
                            domain_file.write(f"add: bullet_at_{x + 1}_{y + 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "UP" in action:
                            if y == 0:
                                continue
                            elif board.is_wall(x, y - 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y - 1}\n")
                            domain_file.write(f"add: bullet_at_{x}_{y - 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "DOWN" in action:
                            if y == board.height - 1:
                                continue
                            elif board.is_wall(x, y + 1):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x}_{y + 1}\n")
                            domain_file.write(f"add: bullet_at_{x}_{y + 1}\n")
                            domain_file.write(f"delete: null\n")
                        elif "LEFT" in action:
                            if x == 0:
                                continue
                            elif board.is_wall(x - 1, y):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x - 1}_{y}\n")
                            domain_file.write(f"add: bullet_at_{x - 1}_{y}\n")
                            domain_file.write(f"delete: null\n")
                        elif "RIGHT" in action:
                            if x == board.width - 1:
                                continue
                            elif board.is_wall(x + 1, y):
                                continue
                            domain_file.write(f"Name: {action}_from_{x}_{y}\n")
                            domain_file.write(f"pre: tank_at_{x}_{y} empty_at_{x + 1}_{y}\n")
                            domain_file.write(f"add: bullet_at_{x + 1}_{y}\n")
                            domain_file.write(f"delete: null\n")

                    # """

        domain_file.close()

    def create_problem_file(self, problem_file_name, board):
        # Create the problem file for the planning graph
        problem_file = open(problem_file_name, 'w')
        problem_file.write("Initial state: ")
        my_tank_num = self.number
        enemy_tank_num = 1 if my_tank_num == 2 else 2
        for x in range(0, board.width):
            for y in range(0, board.height):
                if x == -1 or y == -1 or x == board.width or y == board.height:
                    problem_file.write(f"wall_at_{x}_{y} ")
                elif board.is_wall(x, y):
                    problem_file.write(f"wall_at_{x}_{y} ")
                elif board.tank1.x == x and board.tank1.y == y:
                    if my_tank_num == 1:
                        problem_file.write(f"tank_at_{x}_{y} ")
                        problem_file.write(f"empty_at_{x}_{y} ")
                    else:
                        # problem_file.write(f"enemy_at_{x}_{y} ")
                        problem_file.write(f"empty_at_{x}_{y} ")
                elif board.tank2.x == x and board.tank2.y == y:
                    if my_tank_num == 2:
                        problem_file.write(f"tank_at_{x}_{y} ")
                    else:
                        # problem_file.write(f"enemy_at_{x}_{y} ")
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

        problem_file.close()

    def move(self, _):
        super(PGTank, self).move(_)

        # bad program design. The interface should be the same for all tanks, and the other interface
        # with the update() is better than this.
        # in that case, the update() will do the generate of plan and also call move() and shoot()

        if (self.current_plan_index + 1 >= self.num_turns_to_replan
                or len(self.plan) <= self.num_turns_to_replan):
            self.generate_plan(PLAN_DOMAIN_FILE, PLAN_PROBLEM_FILE)
            if self.decreasing_num_turns_to_replan:
                if self.num_turns_to_replan > 1:
                    self.num_turns_to_replan -= 1

        if self.isplan_uptodate:
            assert self.current_plan_index == 0

        action = self.plan[self.current_plan_index]
        if "MOVE" not in action:
            return False
        # get the direction of the action: "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"
        action_direction = action[action.index("_") + 1:]
        if action in self.get_legal_actions():
            self.current_plan_index += 1
            self.did_move = True
            self.isplan_uptodate = False
            d_x, d_y = GameConstants.STR_TO_VALS[action_direction.lower()]
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
            # get the direction of the action:
            action_direction = action[action.index("_") + 1:]
            if action in self.get_legal_actions():
                d_x, d_y = GameConstants.STR_TO_VALS[action_direction.lower()]
                shot_x, shot_y = self.x + d_x, self.y + d_y
                self.shots -= 1
                self.current_plan_index += 1
                self.isplan_uptodate = False
                self.board.add_bullet(Bullet(self.board, shot_x, shot_y, action_direction))

        self.did_move = False

    def act(self):
        self.move(None)
        self.shoot(None)
