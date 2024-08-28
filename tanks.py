from abc import ABC, abstractmethod
import heapq
from bullet import Bullet
from game_state import GameState

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
        legal_actions = [] #TODO always everything
        if self.shots > 0:
            legal_actions += self.__get_legal_actions_shoot()
        legal_actions += self.__get_legal_actions_move()
        return legal_actions


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


class QLearningTank(Tank):
    def __init__(self, board, x, y, number):
        """
        Initialize an AI-controlled tank using the Q-learning algorithm.

        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        """
        super().__init__(board, x, y, number)
        self.q_table = {}
        self.learning_rate = 0.8  # TODO modify these
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99

        self.best_action = None

    def get_state(self):
        """
        Get the current state of the tank.

        :return: Tuple representing the state.
        """
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        return self.x, self.y, target_tank.x, target_tank.y

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        :param state: Current state.
        :param action: Action taken.
        :return: Q-value for the state-action pair.
        """
        return self.q_table.get((state, action), 0)

    def choose_action(self, state):
        """
        Choose the best action for a given state.

        :param state: Current state.
        :return: Best action for the state.
        """
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.board.actions)
        q_values = [self.get_q_value(state, action) for action in ACTIONS]
        self.best_action = ACTIONS[np.argmax(q_values)]
        return self.best_action

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning algorithm.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state.
        """
        q_value = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, next_action) for next_action in ACTIONS]
        max_q_value = np.max(next_q_values)
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_q_value - q_value)
        self.q_table[(state, action)] = new_q_value

    def move(self, _):
        """
        Move the tank using the Q-learning algorithm to reach the goal.

        :param _: Unused parameter. Needed to match the parent class signature.
        :return: True if move is valid, False otherwise.
        """
        super(QLearningTank, self).move()
        if self.number == 1:
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1

        tank_position = (target_tank.x, target_tank.y)
        state = self.get_state()

        # check if best action starts with MOVE
        if self.best_action.startswith('MOVE'):
            action = self.best_action
        else:
            return False

        next_state = self.board.get_next_state(state, action)
        if self.board.is_valid_move(next_state):
            self.update_q_table(state, action, -1, next_state)
            self.x, self.y = next_state
            return True
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
                # determine direction of bullet according to best action ((dx, dy))
                if self.best_action.startswith('SHOOT'):
                    if self.best_action == 'SHOOT_UP':
                        dx, dy = 0, -1
                    elif self.best_action == 'SHOOT_DOWN':
                        dx, dy = 0, 1
                    elif self.best_action == 'SHOOT_LEFT':
                        dx, dy = -1, 0
                    elif self.best_action == 'SHOOT_RIGHT':
                        dx, dy = 1, 0
                    elif self.best_action == 'SHOOT_UP_LEFT':
                        dx, dy = -1, -1
                    elif self.best_action == 'SHOOT_UP_RIGHT':
                        dx, dy = 1, -1
                    elif self.best_action == 'SHOOT_DOWN_LEFT':
                        dx, dy = -1, 1
                    elif self.best_action == 'SHOOT_DOWN_RIGHT':
                        dx, dy = 1, 1
                else:
                    return
                direction = direction_map.get((dx, dy))
                if direction:
                    self.board.add_bullet(Bullet(self.board, self.x + dx, self.y + dy, direction))
                    self.shots -= 1


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
        self.depth = 1 # TODO change to 3
        self.best_action = None

    def evaluate_game_state(self, game_state):
        """
        Evaluates the game state and returns a score estimating the situation for the player.

        Args:
        - player_position: Tuple (x, y) representing the player's current position.
        - opponent_position: Tuple (x, y) representing the opponent's current position.
        - player_bullets: Integer representing the number of bullets the player has.
        - opponent_bullets: Integer representing the number of bullets the opponent has.
        - bullet_positions: List of tuples [(x, y), ...] representing the positions of all bullets on the board.
        - bullet_directions: List of tuples [(dx, dy), ...] representing the directions of each bullet.
        - weights: Dictionary containing weights for each factor in the heuristic.
        - board_size: Integer representing the size of the board (default is 10 for a 10x10 board).

        Returns:
        - score: A float representing the favorability of the game state for the player. Higher is better.
        """
        board_size = 10

        player_position = (self.x, self.y)
        player_bullets = self.shots
        if self.number == 1:
            opponent_position = (game_state.tank2.x, game_state.tank2.y)
            opponent_bullets = game_state.tank2.shots
        else:
            opponent_position = (game_state.tank1.x, game_state.tank1.y)
            opponent_bullets = game_state.tank1.shots
        bullet_positions = [(bullet.x, bullet.y) for bullet in game_state.bullets]
        bullet_directions = [bullet.direction for bullet in game_state.bullets]

        # TODO: find good weights
        weights = {
            "bullet_count": 10,
            "player_threat": -20,
            "opponent_threat": 20,
            "safe_moves": 5,
            "distance_to_opponent": -1,
            "board_control": 2,
        }

        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def is_bullet_on_trajectory(position, bullet_pos, bullet_dir):
            """
            Checks if the given position is on the trajectory of a bullet considering bounces.
            """
            direction_map = {
                'up': (0, -1),
                'down': (0, 1),
                'left': (-1, 0),
                'right': (1, 0),
                'up_left': (-1, -1),
                'up_right': (1, -1),
                'down_left': (-1, 1),
                'down_right': (1, 1),
            }
            x, y = bullet_pos
            dx, dy = direction_map[bullet_dir]
            for _ in range(10): # TODO change how far we look?
                if (x, y) == position:
                    return True
                # Move bullet and handle bounces
                x += dx
                y += dy
                if x < 0 or x >= board_size:
                    dx = -dx  # Bounce off vertical walls
                    x += 2 * dx
                if y < 0 or y >= board_size:
                    dy = -dy  # Bounce off horizontal walls
                    y += 2 * dy
            return False

        # Initialize raw scores for each factor
        bullet_count_diff = (player_bullets - opponent_bullets) / 5  # Normalize difference by max bullet count
        distance_to_opponent = manhattan_distance(player_position, opponent_position) / (
                    2 * board_size)  # Normalize distance
        player_threat = 0
        opponent_threat = 0

        for i, bullet_pos in enumerate(bullet_positions):
            bullet_dir = bullet_directions[i]

            if is_bullet_on_trajectory(player_position, bullet_pos, bullet_dir):
                player_threat += 1

            if is_bullet_on_trajectory(opponent_position, bullet_pos, bullet_dir):
                opponent_threat += 1

        # Normalize threat levels
        max_threats = len(bullet_positions)  # Maximum number of bullets that could threaten a player
        player_threat_normalized = player_threat / max_threats if max_threats > 0 else 0
        opponent_threat_normalized = opponent_threat / max_threats if max_threats > 0 else 0

        def count_safe_moves(position):
            safe_moves = 0
            possible_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            for move in possible_moves:
                new_position = (position[0] + move[0], position[1] + move[1])
                if 0 <= new_position[0] < board_size and 0 <= new_position[1] < board_size:
                    if not any(
                            is_bullet_on_trajectory(new_position, bullet_pos, bullet_dir) for bullet_pos, bullet_dir in
                            zip(bullet_positions, bullet_directions)):
                        safe_moves += 1
            return safe_moves

        max_moves = 8  # Maximum number of moves a player can have
        player_safe_moves = count_safe_moves(player_position) / max_moves  # Normalize safe moves
        opponent_safe_moves = count_safe_moves(opponent_position) / max_moves  # Normalize safe moves

        def edge_distance(position):
            return min(position[0], board_size - 1 - position[0], position[1], board_size - 1 - position[1])

        max_edge_distance = board_size // 2
        player_edge_distance = edge_distance(player_position) / max_edge_distance
        opponent_edge_distance = edge_distance(opponent_position) / max_edge_distance

        # Compute normalized score components
        normalized_bullet_count = bullet_count_diff
        normalized_threat_diff = opponent_threat_normalized - player_threat_normalized
        normalized_safe_moves_diff = player_safe_moves - opponent_safe_moves
        normalized_distance_to_opponent = -distance_to_opponent  # Closer is better
        normalized_board_control = opponent_edge_distance - player_edge_distance

        # Compute the final score
        score = (
                normalized_bullet_count * weights["bullet_count"] +
                normalized_threat_diff * weights["player_threat"] +
                normalized_safe_moves_diff * weights["safe_moves"] +
                normalized_distance_to_opponent * weights["distance_to_opponent"] +
                normalized_board_control * weights["board_control"]
        )

        return score

    def minimax(self, game_state):
        """
        Minimax algorithm to determine the best move.

        :param state: State of the tank (x, y).
        :param depth: Depth of the search tree.
        :param maximizing_player: Whether the current player is maximizing or minimizing.
        :return: Best value for the current player.
        """

        def max_value(game_state, depth):
            if depth == 0 or game_state.done():
                return self.evaluate_game_state(game_state)
            v = float('-inf')
            for action in game_state.get_legal_actions(1):
                v = max(v, min_value(game_state.generate_successor(1, action), depth))
            return v

        def min_value(game_state, depth):
            if depth == 0 or game_state.done():
                return self.evaluate_game_state(game_state)
            v = float('inf')
            for action in game_state.get_legal_actions(2):
                v = min(v, max_value(game_state.generate_successor(2, action), depth - 1))
            return v

        best_action = None
        best_score = float('-inf')
        for action in ACTIONS:
            score = min_value(game_state.generate_successor(1, action), self.depth)
            if score > best_score:
                best_score = score
                best_action = action
        self.best_action = best_action
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
            target_tank = self.board.tank2
        else:
            target_tank = self.board.tank1
        game_state = GameState(self, target_tank, self.board.bullets)
        action = self.minimax(game_state)
        if action.startswith('MOVE'):
            self.move(action)
        else:
            self.shoot(action)
