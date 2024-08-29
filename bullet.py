from game_colors import GameColors


class Bullet:
    def __init__(self, board, x, y, direction):
        """
        Initialize a bullet.
        
        :param board: Reference to the game board.
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param direction: Direction of movement ('up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right').
        """
        self.board = board  # Reference to the board
        self.x = x  # X coordinate
        self.y = y  # Y coordinate
        self.direction = direction  # Direction of movement
        self.bounces = 0  # Bounce counter
        self.moves = 0  # Move counter

    def move(self):
        """Move the bullet in its direction and handle bounces."""
        self.moves += 1
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
        if self.direction in directions:
            dx, dy = directions[self.direction]
            self.board.update_position(self.x, self.y, GameColors.BOARD)
            self.x += dx
            self.y += dy

        if self.x < 0 or self.x >= self.board.size or self.y < 0 or self.y >= self.board.size:
            if self.bounces < 2:
                self.bounces += 1
                bounce_map = {
                    'up': 'down',
                    'down': 'up',
                    'left': 'right',
                    'right': 'left',
                    'up_left': 'down_right',
                    'up_right': 'down_left',
                    'down_left': 'up_right',
                    'down_right': 'up_left'
                }
                self.direction = bounce_map[self.direction]
            else:
                self.board.remove_bullet(self)
                return
        else:
            self.board.move_bullet(self, self.x, self.y)
        self.board.update_position(self.x, self.y, GameColors.BULLET)
