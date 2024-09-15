class GameConstants:
    # Colors
    BOARD = 'dim gray'
    TANK1 = 'RoyalBlue1'
    TANK2 = 'maroon'
    BULLET = 'black'
    BOUNCED_BULLET = BULLET  # Color of a bullet that has bounced off a wall. May change if desired
    OUTLINE = 'seashell2'
    WALL = 'saddle brown'

    # Settings
    DELAY_MS = 1000  # Delay in milliseconds for NPC actions
    BOARD_WIDTH = 10  # Width of the game board
    BOARD_HEIGHT = 10  # Height of the game board
    MAX_TURNS = 100  # Maximum number of turns before a draw
    BEGINNING_SHOTS = 3  # Number of shots each tank starts with
    MAX_SHOTS = 5  # Maximum number of shots a tank can have
    MAX_BOUNCES = 4  # Maximum number of bounces for a bullet

    # Helpful Datastructures
    ACTIONS = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP_LEFT', 'MOVE_UP_RIGHT', 'MOVE_DOWN_LEFT',
               'MOVE_DOWN_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP_LEFT',
               'SHOOT_UP_RIGHT', 'SHOOT_DOWN_LEFT', 'SHOOT_DOWN_RIGHT']

    VALS_TO_STR = {
        (0, -1): 'up',
        (0, 1): 'down',
        (-1, 0): 'left',
        (1, 0): 'right',
        (-1, -1): 'up_left',
        (1, -1): 'up_right',
        (-1, 1): 'down_left',
        (1, 1): 'down_right'
    }

    STR_TO_VALS = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0),
        'up_left': (-1, -1),
        'up_right': (1, -1),
        'down_left': (-1, 1),
        'down_right': (1, 1),
    }
