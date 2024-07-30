import tkinter as tk
import heapq
from abc import ABC, abstractmethod

DELAY_MS = 500  # Delay in milliseconds for NPC actions


class Board:
    def __init__(self, size, main_window, delay=False):
        """
        Initialize the game board.
        
        :param size: The size of the board (number of tiles in one dimension).
        :param main_window: The main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        """
        self.size = size  # Board size
        self.grid = [[' ' for _ in range(size)] for _ in range(size)]  # Board grid
        self.main_window = main_window  # Main window reference
        self.window = tk.Toplevel(self.main_window)  # Game window
        self.window.title("Ricochet Rumble")
        self.canvas = tk.Canvas(self.window, width=500, height=500)  # Canvas for drawing
        self.canvas.pack()
        self.cell_size = 500 // size  # Size of each cell
        self.tank1 = None  # Reference to tank 1
        self.tank2 = None  # Reference to tank 2
        self.bullets = []  # List of bullets
        self.mode = 'move'  # Current mode (move or shoot)
        self.turns = 0  # Turn counter
        self.message_label = tk.Label(self.window, text="")  # Label for messages
        self.message_label.pack()
        self.mode_label = tk.Label(self.window, text=f"Mode: {self.mode}")  # Label for mode
        self.mode_label.pack()
        self.quit_button = tk.Button(self.window, text="Quit", command=self.quit_game)  # Quit button
        self.quit_button.pack()
        self.delay = delay  # Delay flag

    def draw_grid(self):
        """Draw the grid on the canvas."""
        for i in range(self.size):
            for j in range(self.size):
                self.canvas.create_rectangle(
                    i * self.cell_size, j * self.cell_size,
                    (i + 1) * self.cell_size, (j + 1) * self.cell_size,
                    fill='white', outline='black'
                )

    def place_tank(self, tank, number):
        """
        Place a tank on the board.
        
        :param tank: Tank object to place.
        :param number: Tank number (1 or 2).
        """
        if number == 1:
            self.tank1 = tank
            color = 'blue'
        else:
            self.tank2 = tank
            color = 'red'
        self.update_position(tank.x, tank.y, color)

    def update_position(self, x, y, color):
        """
        Update the position of an object on the board.
        
        :param x: X coordinate.
        :param y: Y coordinate.
        :param color: Color of the object.
        """
        self.canvas.create_rectangle(
            x * self.cell_size, y * self.cell_size,
            (x + 1) * self.cell_size, (y + 1) * self.cell_size,
            fill=color, outline='black'
        )

    def move_tank(self, tank, new_x, new_y, number):
        """
        Move a tank to a new position.
        
        :param tank: Tank object to move.
        :param new_x: New X coordinate.
        :param new_y: New Y coordinate.
        :param number: Tank number (1 or 2).
        :return: True if move is valid, False otherwise.
        """
        if self.is_valid_move(new_x, new_y):
            color = 'blue' if number == 1 else 'red'
            self.update_position(tank.x, tank.y, 'white')
            tank.x, tank.y = new_x, new_y
            self.update_position(new_x, new_y, color)
            return True
        else:
            self.show_message("Invalid move!")
            return False

    def add_bullet(self, bullet):
        """
        Add a bullet to the board.
        
        :param bullet: Bullet object to add.
        """
        self.bullets.append(bullet)
        self.update_position(bullet.x, bullet.y, 'black')

    def move_bullet(self, bullet, new_x, new_y):
        """
        Move a bullet to a new position.
        
        :param bullet: Bullet object to move.
        :param new_x: New X coordinate.
        :param new_y: New Y coordinate.
        """
        self.update_position(bullet.x, bullet.y, 'white')
        bullet.x, bullet.y = new_x, new_y
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.update_position(new_x, new_y, 'black')

    def remove_bullet(self, bullet):
        """
        Remove a bullet from the board.
        
        :param bullet: Bullet object to remove.
        """
        self.update_position(bullet.x, bullet.y, 'white')
        self.bullets.remove(bullet)

    def check_bullet_collisions(self):
        """
        Check for bullet collisions with tanks.
        
        :return: True if a collision occurs, False otherwise.
        """
        for bullet in self.bullets:
            if bullet.x == self.tank1.x and bullet.y == self.tank1.y:
                self.show_message("Tank1 hit!")
                self.end_game("Tank2 wins!")
                return True
            elif bullet.x == self.tank2.x and bullet.y == self.tank2.y:
                self.show_message("Tank2 hit!")
                self.end_game("Tank1 wins!")
                return True
        return False

    def is_valid_move(self, x, y):
        """
        Check if a move is valid.
        
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the move is valid, False otherwise.
        """
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[y][x] == ' '

    def show_message(self, message):
        """
        Display a message.
        
        :param message: Message to display.
        """
        self.message_label.config(text=message)

    def update_mode(self):
        """Update the mode label."""
        self.mode_label.config(text=f"Mode: {self.mode.capitalize()}")

    def update_bullets(self):
        """Update the positions of all bullets."""
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.moves >= 10:
                self.remove_bullet(bullet)

    def end_game(self, result_message):
        """
        End the game and show the result.
        
        :param result_message: Message to display.
        """
        self.window.destroy()
        EndScreen(self.main_window, result_message)

    def quit_game(self):
        """Quit the game."""
        self.window.destroy()
        self.main_window.quit()

    def delay_action(self, func):
        """
        Execute an action with an optional delay.
        
        :param func: Function to execute.
        """
        if self.delay:
            self.window.after(DELAY_MS, func)
        else:
            func()


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
        self.shots = 0  # Shot counter
        self.number = number  # Tank number
        board.place_tank(self, number)

    @abstractmethod
    def move(self, direction):
        """Move the tank in a specified direction."""
        pass

    @abstractmethod
    def shoot(self, direction):
        """Shoot a bullet in a specified direction."""
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
        if self.shots < 3:
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
                bullet = Bullet(self.board, self.x + dx, self.y + dy, direction)
                self.board.add_bullet(bullet)
                self.shots += 1
        else:
            self.board.show_message("You can't shoot yet!")


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
        if self.shots < 3:
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
                    self.shots += 1


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
        self.board.update_position(self.x, self.y, 'white')
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


class MainMenu:
    def __init__(self, main_window):
        """
        Initialize the main menu.
        
        :param main_window: Reference to the main Tkinter window.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # Menu window
        self.window.pack()

        self.tank1_var = tk.StringVar(value="Player")  # Tank 1 type
        self.tank2_var = tk.StringVar(value="A*")  # Tank 2 type

        tank1_label = tk.Label(self.window, text="Tank 1:")  # Tank 1 label
        tank1_label.pack(pady=10)
        tank1_options = tk.OptionMenu(self.window, self.tank1_var, "Player", "A*")  # Tank 1 options
        tank1_options.pack(pady=10)

        tank2_label = tk.Label(self.window, text="Tank 2:")  # Tank 2 label
        tank2_label.pack(pady=10)
        tank2_options = tk.OptionMenu(self.window, self.tank2_var, "Player", "A*")  # Tank 2 options
        tank2_options.pack(pady=10)

        start_button = tk.Button(self.window, text="Start Game", command=self.start_game)  # Start game button
        start_button.pack(pady=10)

        self.delay_var = tk.BooleanVar()  # Delay option
        delay_checkbox = tk.Checkbutton(self.window, text="Enable delay", variable=self.delay_var)  # Delay checkbox
        delay_checkbox.pack(pady=10)

        quit_button = tk.Button(self.window, text="Quit", command=self.quit_game)  # Quit button
        quit_button.pack(pady=10)

    def start_game(self):
        """Start the game with selected options."""
        self.window.pack_forget()
        Game(self.main_window, self.delay_var.get(), self.tank1_var.get(), self.tank2_var.get())

    def quit_game(self):
        """Quit the game."""
        self.main_window.quit()


class Game:
    def __init__(self, main_window, delay, tank1_type, tank2_type):
        """
        Initialize the game.
        
        :param main_window: Reference to the main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        :param tank1_type: Type of tank 1 ('Player' or 'A*').
        :param tank2_type: Type of tank 2 ('Player' or 'A*').
        """
        self.board = Board(10, main_window, delay)  # Game board
        self.board.draw_grid()
        self.main_window = main_window  # Main window reference
        self.main_window.bind('<KeyRelease>', self.handle_key_release)  # Key release handler
        self.main_window.bind('<KeyPress>', self.handle_key_press)  # Key press handler
        self.last_keys = []  # List of last pressed keys
        self.player_action_done = False  # Flag to track player action

        self.tank1 = self.create_tank(tank1_type, 0, 0, 1)  # Tank 1
        self.tank2 = self.create_tank(tank2_type, 9, 9, 2)  # Tank 2

        self.current_tank = self.tank1  # Current tank
        self.turns = 0  # Turn counter

    def create_tank(self, tank_type, x, y, number):
        """
        Create a tank based on type.
        
        :param tank_type: Type of tank ('Player' or 'A*').
        :param x: Initial X coordinate.
        :param y: Initial Y coordinate.
        :param number: Tank number (1 or 2).
        :return: Tank object.
        """
        if tank_type == "Player":
            return PlayerTank(self.board, x, y, number)
        elif tank_type == "A*":
            return AStarTank(self.board, x, y, number)
        # Future tank types can be added here

    def handle_key_press(self, event):
        """
        Handle key press events.

        :param event: Key press event.
        """
        if event.keysym not in self.last_keys:
            self.last_keys.append(event.keysym)

    def handle_key_release(self, event):
        """
        Handle key release events.

        :param event: Key release event.
        """
        if event.keysym == 'm':
            self.board.mode = 'shoot' if self.board.mode == 'move' else 'move'
            self.board.update_mode()
            return

        if self.player_action_done or not isinstance(self.current_tank, PlayerTank):
            return

        direction_map = {
            'w': 'up', 's': 'down', 'a': 'left', 'd': 'right',
            'q': 'up_left', 'e': 'up_right', 'z': 'down_left', 'c': 'down_right'
        }
        valid_action = False
        if self.board.mode == 'move':
            if event.keysym in direction_map:
                valid_action = self.current_tank.move(direction_map[event.keysym])
            else:
                self.board.show_message("Invalid key!")
        elif self.board.mode == 'shoot':
            if event.keysym in direction_map:
                self.current_tank.shoot(direction_map[event.keysym])
                valid_action = True
            else:
                self.board.show_message("Invalid key!")

        if valid_action:
            self.player_action_done = True
            self.last_keys.clear()
            if not self.board.check_bullet_collisions():
                self.switch_turn()

    def switch_turn(self):
        """Switch turns between tanks."""
        self.turns += 1
        if self.current_tank == self.tank1:
            self.current_tank = self.tank2
        else:
            self.current_tank = self.tank1

        if isinstance(self.current_tank, AStarTank):
            self.board.delay_action(self.npc_act)
        else:
            self.player_action_done = False

    def npc_act(self):
        """Perform action for NPC tank."""
        self.current_tank.move(None)  # Move towards the target tank
        self.current_tank.shoot(None)  # Shoot if possible
        self.board.update_bullets()
        if not self.board.check_bullet_collisions():
            self.switch_turn()


class EndScreen:
    def __init__(self, main_window, result_message):
        """
        Initialize the end screen.
        
        :param main_window: Reference to the main Tkinter window.
        :param result_message: Result message to display.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # End screen window
        self.window.pack()

        result_label = tk.Label(self.window, text=result_message)  # Result message label
        result_label.pack(pady=10)

        replay_button = tk.Button(self.window, text="Replay", command=self.replay_game)  # Replay button
        replay_button.pack(pady=10)

        quit_button = tk.Button(self.window, text="Quit", command=self.quit_game)  # Quit button
        quit_button.pack(pady=10)

    def replay_game(self):
        """Replay the game."""
        self.window.pack_forget()
        MainMenu(self.main_window)

    def quit_game(self):
        """Quit the game."""
        self.main_window.quit()


def main():
    """Main function to start the game."""
    main_window = tk.Tk()
    main_window.title("Ricochet Rumble")
    MainMenu(main_window)
    main_window.mainloop()


if __name__ == '__main__':
    main()
