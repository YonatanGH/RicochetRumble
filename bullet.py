from game_constants import GameConstants


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

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

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
        if self.direction not in directions:
            raise ValueError('Invalid direction')

        dx, dy = directions[self.direction]

        if self.board.is_wall(self.x + dx, self.y + dy):
            if self.bounces < GameConstants.MAX_BOUNCES:
                self.bounces += 1
                bounce_map = {
                    'up': 'down',
                    'down': 'up',
                    'left': 'right',
                    'right': 'left',
                    'up_left': 'down_right',
                    'up_right': 'down_left',
                    'down_left': 'up_right',
                    'down_right': 'up_left',
                    'up_left_vertical_wall': 'up_right',
                    'up_right_vertical_wall': 'up_left',
                    'down_left_vertical_wall': 'down_right',
                    'down_right_vertical_wall': 'down_left',
                    'up_left_horizontal_wall': 'down_left',
                    'up_right_horizontal_wall': 'down_right',
                    'down_left_horizontal_wall': 'up_left',
                    'down_right_horizontal_wall': 'up_right'
                }
                # if the bullet is moving diagonally, change the direction to the opposite diagonal,
                # according to the wall it hit. If it is a corner, then use the bounce_map.
                if dx == 0 or dy == 0:
                    self.direction = bounce_map[self.direction]
                else:  # diagonal - TODO: there exists an option of a bullet in a 1-width path, that gets stuck
                    # if (self.board.is_wall(self.x + dx, self.y) and self.board.is_wall(self.x, self.y + dy) or
                    #         (not self.board.is_wall(self.x + dx, self.y) and not self.board.is_wall(self.x,
                    #                                                                                 self.y + dy))):
                    if not (self.board.is_wall(self.x + dx, self.y) ^ self.board.is_wall(self.x, self.y + dy)):
                        self.direction = bounce_map[self.direction]
                    elif self.board.is_wall(self.x, self.y + dy):
                        self.direction = bounce_map[self.direction + '_horizontal_wall']
                    elif self.board.is_wall(self.x + dx, self.y):
                        self.direction = bounce_map[self.direction + '_vertical_wall']
                # self.direction = bounce_map[self.direction]
                self.board.update_position(self.x, self.y, GameConstants.BOUNCED_BULLET)
            else:
                self.board.remove_bullet(self)
        # if bullets collide with each other, remove both bullets
        elif self.board.is_bullet(self.x + dx, self.y + dy):
            self.board.remove_bullet(self)
            self.board.remove_bullet(self.board.get_bullet(self.x + dx, self.y + dy))
        else:
            self.board.update_position(self.x, self.y, GameConstants.BOARD)
            self.x += dx
            self.y += dy
            self.board.move_bullet(self, self.x, self.y)
            self.board.update_position(self.x, self.y, GameConstants.BULLET)
