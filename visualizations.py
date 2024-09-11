import tkinter as tk

from game_colors import GameColors
from maze import generate_spacious_maze
from tanks import PlayerTank, AStarTank, MinimaxTank, QLearningTank, ExpectimaxTank, PGTank, RandomTank
from tournament_league import TournamentLeague, ResultsTracker, MegaTournament

DELAY_MS = 1000  # Delay in milliseconds for NPC actions
BOARD_WIDTH = 10  # Width of the game board
BOARD_HEIGHT = 10  # Height of the game board
MAX_TURNS = 100  # Maximum number of turns before a draw


# -------------------------------------- Main Menu -------------------------------------- #
class MainMenu:
    def __init__(self, main_window, qchoice1="None", qchoice2="None"):
        """
        Initialize the main menu.
        
        :param main_window: Reference to the main Tkinter window.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # Menu window
        self.window.pack()

        self.options = ["Player", "A*", "Planning-Graph", "Minimax", "Expectimax", "Q-Learning", "Random"]  # Tanks

        self.qchoice1 = qchoice1
        self.qchoice2 = qchoice2

        self.tank1_var = tk.StringVar(value="Player")  # Tank 1 type
        self.tank2_var = tk.StringVar(value="A*")  # Tank 2 type

        tank1_label = tk.Label(self.window, text="Tank 1:")  # Tank 1 label
        tank1_label.pack(pady=10)
        tank1_options = tk.OptionMenu(self.window, self.tank1_var, *self.options)  # Tank 1 options
        tank1_options.pack(pady=10)


        tank2_label = tk.Label(self.window, text="Tank 2:")  # Tank 2 label
        tank2_label.pack(pady=10)
        tank2_options = tk.OptionMenu(self.window, self.tank2_var, *self.options)  # Tank 2 options
        tank2_options.pack(pady=10)

        start_button = tk.Button(self.window, text="Start Game", command=self.start_game)  # Start game button
        start_button.pack(pady=10)

        manual_button = tk.Button(self.window, text="Tank Manual", command=self.show_tank_manual)  # Tank Manual button
        manual_button.pack(pady=10)

        settings_button = tk.Button(self.window, text="Q-Settings", command=self.show_settings)  # Tank Manual button
        settings_button.pack(pady=10)

        self.delay_var = tk.BooleanVar(value=True)  # Delay variable
        delay_checkbox = tk.Checkbutton(self.window, text="Enable delay", variable=self.delay_var)  # Delay checkbox
        delay_checkbox.pack(pady=10)

        # Tournament options
        tournament_button = tk.Button(self.window, text="Tournaments", command=self.tournament_menu)  # Start tournament button
        tournament_button.pack(pady=10)

        quit_button = tk.Button(self.window, text="Quit", command=self.quit_game)  # Quit button
        quit_button.pack(pady=10)

    def start_game(self):
        """Start the game with selected options."""
        self.window.pack_forget()
        if type(self.qchoice1) == str:
            self.qchoice1 = tk.StringVar(value="None")
        if type(self.qchoice2) == str:
            self.qchoice2 = tk.StringVar(value="None")
        Game(self.main_window, self.delay_var.get(), self.tank1_var.get(), self.tank2_var.get(), self.qchoice1.get(), self.qchoice2.get())

    def tournament_menu(self):
        """Start the tournament menu."""
        self.window.pack_forget()
        TournamentMenu(self.main_window)
    def start_tournament(self, tank1, tank2, num_games, num_visualized):
        """Start the tournament with selected options."""
        self.window.pack_forget()
        if type(self.qchoice1) == str:
            self.qchoice1 = tk.StringVar(value="None")
        if type(self.qchoice2) == str:
            self.qchoice2 = tk.StringVar(value="None")
        TournamentLeague(tank1, tank2, self.main_window, self.qchoice1.get(), self.qchoice2.get(), num_games=num_games, amount_of_visualizations=num_visualized)

    def start_all_out_tournament(self, num_games):
        MegaTournament(["A*", "Planning-Graph", "Minimax", "Expectimax",
                        "Q-Learning|A*",
                        "Q-Learning|Planning-Graph",
                        "Q-Learning|Minimax",
                        "Q-Learning|Expectimax",
                        "Random"],self.main_window, num_games)

    def show_tank_manual(self):
        """Show the tank manual."""
        self.window.pack_forget()
        TankManual(self.main_window)

    def show_settings(self):
        """Show the settings."""
        self.window.pack_forget()
        Settings(self.main_window)

    def quit_game(self):
        """Quit the game."""
        self.main_window.quit()


# -------------------------------------- Tank Manual -------------------------------------- #

class TankManual:
    def __init__(self, main_window):
        """
        Initialize the tank manual screen.

        :param main_window: Reference to the main Tkinter window.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # Manual window
        self.window.pack()

        # Manual content # TODO: add yours!
        manual_text = (
            "Tank Manual:\n\n"
            "PlayerTank:\n"
            "- You control this tank with the 'w', 'a', 's', 'd' keys to move, "
            "and 'q', 'e', 'z', 'c' for diagonal movement.\n"
            "- Press 'm' to switch between move and shoot modes.\n"
            "- In shoot mode, use the same keys to shoot in different directions.\n"
            "- The tank can hold up to 3 bullets at a time.\n\n"

            "A* Tank:\n"
            "- This AI-controlled tank uses the A* algorithm to find and move towards the player's tank.\n"
            "- It automatically shoots at the player when in range.\n\n"

            "Random Tank:\n"
            "- This tank has a chance of 20% to shoot and 80% to move. Its actions are valid, but fully random.\n"
            
            "Planning-Graph Tank:\n"
            "- This AI-controlled tank uses the Planning-Graph algorithm to determine the best action.\n"
            
            "Minimax Tank:\n"
            "- This AI-controlled tank uses the Minimax algorithm to determine the best action.\n"

            "Expectimax Tank:\n"
            "- This AI-controlled tank uses the Expectimax algorithm to determine the best action.\n"

            "Q-Learning Tank:\n"
            "- This AI-controlled tank uses the Q-Learning algorithm to learn about the game in advance.\n"
            "- It uses the learned information to determine the best action in each turn.\n"
            "- The tank will learn to play against the tank chosen in the settings.\n"
        )

        manual_label = tk.Label(self.window, text=manual_text, justify="left")
        manual_label.pack(pady=10)

        back_button = tk.Button(self.window, text="Back to Menu", command=self.back_to_menu)  # Back to menu button
        back_button.pack(pady=10)

    def back_to_menu(self):
        """Return to the main menu."""
        self.window.pack_forget()
        MainMenu(self.main_window)


# -------------------------------------- Settings -------------------------------------- #

class Settings:
    def __init__(self, main_window):
        """
        Initialize the tank manual screen.

        :param main_window: Reference to the main Tkinter window.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # Manual window
        self.window.pack()

        self.options = ["None", "A*", "Planning-Graph", "Minimax", "Expectimax", "Random"]

        self.choice1 = tk.StringVar(value="None")
        self.choice2 = tk.StringVar(value="None")

        tank1_label = tk.Label(self.window, text="Pre-training for tank 1:")
        tank1_label.pack(pady=10)
        tank1_options = tk.OptionMenu(self.window, self.choice1, *self.options)
        tank1_options.pack(pady=10)

        tank2_label = tk.Label(self.window, text="Pre-training for tank 2:")
        tank2_label.pack(pady=10)
        tank2_options = tk.OptionMenu(self.window, self.choice2, *self.options)
        tank2_options.pack(pady=10)

        back_button = tk.Button(self.window, text="Back to Menu", command=self.back_to_menu)  # Back to menu button
        back_button.pack(pady=10)

    def back_to_menu(self):
        """Return to the main menu."""
        self.window.pack_forget()
        MainMenu(self.main_window, self.choice1.get(), self.choice2.get())

    def get_choices(self):
        return self.choice1, self.choice2


# -------------------------------------- Tournament Manual -------------------------------------- #
class TournamentMenu:
    def __init__(self, main_window):
        """
        Initialize the tank manual screen.

        :param main_window: Reference to the main Tkinter window.
        """
        self.main_window = main_window  # Main window reference
        self.window = tk.Frame(main_window)  # Manual window
        self.window.pack()

        self.options = ["Player", "A*", "Planning-Graph", "Minimax", "Expectimax", "Q-Learning", "Random"]

        self.choice1 = tk.StringVar(value="A*")
        self.choice2 = tk.StringVar(value="Random")

        tank1_label = tk.Label(self.window, text="tank 1:")
        tank1_label.pack(pady=10)
        tank1 = tk.OptionMenu(self.window, self.choice1, *self.options)
        tank1.pack(pady=10)

        tank2_label = tk.Label(self.window, text="Tank 2:")
        tank2_label.pack(pady=10)
        tank2 = tk.OptionMenu(self.window, self.choice2, *self.options)
        tank2.pack(pady=10)

        # number of games
        self.num_games = tk.StringVar(value="1")
        num_games_label = tk.Label(self.window, text="Number of games:")
        num_games_label.pack(pady=10)
        num_games_options = tk.OptionMenu(self.window, self.num_games,"1", "3", "5", "10", "25", "50", "100")
        num_games_options.pack(pady=10)

        # number of games to visualize
        self.num_visualized = tk.StringVar(value="0")
        num_games_label = tk.Label(self.window, text="Number of visualized games:")
        num_games_label.pack(pady=10)
        num_games_options = tk.OptionMenu(self.window, self.num_visualized,"0","1", "3", "5", "10", "25", "50", "100")
        num_games_options.pack(pady=10)

        start_button = tk.Button(self.window, text="Start Tournament", command=self.start_tournament)  # Start game button
        start_button.pack(pady=10)

        all_out_tournament_button = tk.Button(self.window, text="Start All-Out Tournament", command=self.start_all_out_tournament)  # Start game button
        all_out_tournament_button.pack(pady=10)

        back_button = tk.Button(self.window, text="Back to Menu", command=self.back_to_menu)  # Back to menu button
        back_button.pack(pady=10)

    def start_tournament(self):
        """Start the tournament with selected options."""
        self.window.pack_forget()
        main_menu = MainMenu(self.main_window)
        main_menu.start_tournament(self.choice1.get(), self.choice2.get(), int(self.num_games.get()), int(self.num_visualized.get()))
        # TournamentLeague(self.choice1.get(), self.choice2.get(), self.main_window, int(self.num_games.get()))

    def start_all_out_tournament(self):
        """Start the tournament with selected options."""
        self.window.pack_forget()
        main_menu = MainMenu(self.main_window)
        main_menu.start_all_out_tournament(int(self.num_games.get()))

    def back_to_menu(self):
        """Return to the main menu."""
        self.window.pack_forget()
        MainMenu(self.main_window, self.choice1.get(), self.choice2.get())

    def get_choices(self):
        return self.choice1, self.choice2


# -------------------------------------- End Screen -------------------------------------- #

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


# -------------------------------------- Board -------------------------------------- #

class Board:
    def __init__(self, width, height, main_window, delay=False, result_tracker=None, enable_endscreen=True):
        """
        Initialize the game board.
        
        :param width: Width of the board.
        :param height: Height of the board.
        :param main_window: The main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        """
        self.result_tracker = result_tracker
        self.enable_endscreen = enable_endscreen
        self.width = width  # Width of the board
        self.height = height  # Height of the board

        self.grid = [[GameColors.BOARD for _ in range(width)] for _ in range(height)]  # Grid of colors
        self.generate_maze()

        self.main_window = main_window  # Main window reference
        self.window = tk.Toplevel(self.main_window)  # Game window
        self.window.title("Ricochet Rumble")
        self.canvas = tk.Canvas(self.window, width=500, height=500)  # Canvas for drawing
        self.canvas.pack()

        self.cell_size = 500 // width  # Cell size

        self.tank1 = None  # Reference to tank 1
        self.tank2 = None  # Reference to tank 2
        self.bullets = []  # List of bullets
        self.mode = 'move'  # Current mode (move or shoot)
        self.turns = 0  # Turn counter
        self.message_label = tk.Label(self.window, text="")  # Label for messages
        self.message_label.pack()

        # Bullet counts
        self.bullet_label1 = tk.Label(self.window, text="Tank 1 Bullets: 0")  # Bullet count label for Tank 1
        self.bullet_label1.pack(side="left")

        self.bullet_label2 = tk.Label(self.window, text="Tank 2 Bullets: 0")  # Bullet count label for Tank 2
        self.bullet_label2.pack(side="right")

        # Mode indicators
        self.mode_label1 = tk.Label(self.window, text="Mode: Move", fg=GameColors.TANK1)  # Mode label for Tank 1
        self.mode_label2 = tk.Label(self.window, text="Mode: Move", fg=GameColors.TANK2)  # Mode label for Tank 2

        self.quit_button = tk.Button(self.window, text="Quit", command=self.quit_game)  # Quit button
        self.quit_button.pack()
        self.delay = delay  # Delay flag

    def draw_grid(self):
        """Draw the grid on the canvas"""
        for y in range(self.height):
            for x in range(self.width):
                self.canvas.create_rectangle(
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                    fill=self.grid[y][x], outline=GameColors.OUTLINE
                )

    def place_tank(self, tank, number):
        """
        Place a tank on the board.
        
        :param tank: Tank object to place.
        :param number: Tank number (1 or 2).
        """
        if number == 1:
            self.tank1 = tank
            color = GameColors.TANK1
        else:
            self.tank2 = tank
            color = GameColors.TANK2
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
            fill=color, outline=GameColors.OUTLINE
        )
        # Update the grid
        self.grid[y][x] = color

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
            color = GameColors.TANK1 if number == 1 else GameColors.TANK2
            self.update_position(tank.x, tank.y, GameColors.BOARD)  # Repaint old position
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
        if self.is_valid_move(bullet.x, bullet.y):
            self.bullets.append(bullet)
            self.update_position(bullet.x, bullet.y, GameColors.BULLET)
            return True
        self.show_message("Invalid move!")
        return False

    def move_bullet(self, bullet, new_x, new_y):
        """
        Move a bullet to a new position.
        
        :param bullet: Bullet object to move.
        :param new_x: New X coordinate.
        :param new_y: New Y coordinate.
        """
        self.update_position(bullet.x, bullet.y, GameColors.BOARD)  # Repaint old position
        bullet.x, bullet.y = new_x, new_y
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.update_position(new_x, new_y, GameColors.BULLET)

    def remove_bullet(self, bullet):
        """
        Remove a bullet from the board.
        
        :param bullet: Bullet object to remove.
        """
        self.update_position(bullet.x, bullet.y, GameColors.BOARD)  # Repaint old position with brown
        self.bullets.remove(bullet)

    def check_bullet_collisions(self):
        """
        Check for bullet collisions with tanks.

        :return: True if a collision occurs, False otherwise.
        """
        for bullet in self.bullets:
            if bullet.x == self.tank1.x and bullet.y == self.tank1.y:
                self.end_game(f"Tank 2 ({self.tank2.__class__.__name__}) wins!", winner=2)
                return True
            elif bullet.x == self.tank2.x and bullet.y == self.tank2.y:
                self.end_game(f"Tank 1 ({self.tank1.__class__.__name__}) wins!", winner=1)
                return True
        return False

    def is_valid_move(self, x, y):
        """
        Check if a move is valid.

        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the move is valid, False otherwise.
        """
        return 0 <= x < self.width and 0 <= y < self.height and (self.grid[y][x] == GameColors.BOARD or
                                                                 self.grid[y][x] == GameColors.TANK1 or
                                                                 self.grid[y][x] == GameColors.TANK2)

    def show_message(self, message):
        """
        Display a message.

        :param message: Message to display.
        """
        self.message_label.config(text=message)

    def update_mode(self, current_tank):
        """Update the mode label based on the current tank's type."""
        # instead of writing tank 1 and tank 2, we can write their types

        if isinstance(self.tank1, PlayerTank):
            self.mode_label1.config(text=f"Mode: {self.mode.capitalize()}")
            self.mode_label1.pack(side="left", pady=(0, 10))

        if isinstance(self.tank2, PlayerTank):
            self.mode_label2.config(text=f"Mode: {self.mode.capitalize()}")
            self.mode_label2.pack(side="right", pady=(0, 10))

    def update_bullets(self):
        """Update the positions of all bullets."""
        for bullet in self.bullets:
            if bullet.moves > 0:
                bullet.move()
            else:
                bullet.moves += 1
            if bullet.moves >= 10:
                if bullet:
                    self.remove_bullet(bullet)

    def end_game(self, result_message, winner=None):
        """
        End the game and show the result.

        :param result_message: Message to display.
        """
        if self.result_tracker != None and winner != None:
            if winner == 1:
                self.result_tracker.add_tank1_win()
            elif winner == 2:
                self.result_tracker.add_tank2_win()
            else:
                self.result_tracker.add_draw()
        self.window.destroy()
        if self.enable_endscreen:
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

    def update_bullet_count(self):
        """Update the bullet count display for both tanks."""
        tank_type1 = self.tank1.__class__.__name__ if self.tank1 else "Tank 1"
        tank_type2 = self.tank2.__class__.__name__ if self.tank2 else "Tank 2"
        self.bullet_label1.config(text=f"Tank 1 ({tank_type1}) Bullets: {self.tank1.shots}")
        self.bullet_label2.config(text=f"Tank 2 ({tank_type2}) Bullets: {self.tank2.shots}")

    def generate_maze(self):
        """Generate the maze and update the grid."""
        maze = generate_spacious_maze(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                if maze[y][x] == 'W':
                    self.grid[y][x] = GameColors.WALL  # Use wall color for walls
                else:
                    self.grid[y][x] = GameColors.BOARD  # Use board color for paths

    def is_wall(self, x, y):
        """
        Check if a position is a wall.
        
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the position is a wall, False otherwise.
        """
        return x < 0 or x >= self.width or y < 0 or y >= self.height or self.grid[y][x] == GameColors.WALL

    def get_bullet(self, x, y):
        """
        Get the bullet at a position.
        
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: Bullet object if found, None otherwise.
        """
        for bullet in self.bullets:
            if bullet.x == x and bullet.y == y:
                return bullet
        return None

    def is_bullet(self, x, y):
        """
        Check if a position has a bullet.
        
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the position has a bullet, False otherwise.
        """
        return self.get_bullet(x, y) is not None


# -------------------------------------- Game -------------------------------------- #

class Game:
    def __init__(self, main_window, delay, tank1_type, tank2_type, qmode1="None", qmode2="None", result_tracker=None, enable_endscreen=True):
        """
        Initialize the game.
        
        :param main_window: Reference to the main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        :param tank1_type: Type of tank 1 ('Player' or 'A*').
        :param tank2_type: Type of tank 2 ('Player' or 'A*').
        """
        self.result_tracker = result_tracker
        self.enable_endscreen = enable_endscreen
        self.board = Board(BOARD_WIDTH, BOARD_HEIGHT, main_window, delay, result_tracker=result_tracker, enable_endscreen=self.enable_endscreen)  # Game board
        self.board.draw_grid()
        self.main_window = main_window  # Main window reference
        self.main_window.bind('<KeyRelease>', self.handle_key_release)  # Key release handler
        self.main_window.bind('<KeyPress>', self.handle_key_press)  # Key press handler
        self.last_keys = []  # List of last pressed keys
        self.player_action_done = False  # Flag to track player action

        self.tank1 = self.create_tank(tank1_type, 0, 0, 1, qmode1)  # Tank 1
        self.tank2 = self.create_tank(tank2_type, BOARD_WIDTH - 1, BOARD_HEIGHT - 1, 2, qmode2)  # Tank 2

        self.current_tank = self.tank1  # Current tank
        self.turns = 0  # Turn counter

        if not isinstance(self.current_tank, PlayerTank):
            # make the first move for the NPC tank
            self.board.delay_action(self.npc_act)

    def create_tank(self, tank_type, x, y, number, mode):
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
        elif tank_type == "Random":
            return RandomTank(self.board, x, y, number)
        elif tank_type == "A*":
            return AStarTank(self.board, x, y, number)
        # Future tank types can be added here
        elif tank_type == "Planning-Graph":
            return PGTank(self.board, x, y, number)
        elif tank_type == "Minimax":
            return MinimaxTank(self.board, x, y, number)
        elif tank_type == "Expectimax":
            return ExpectimaxTank(self.board, x, y, number)
        elif tank_type == "Q-Learning":
            if mode == "None":
                return QLearningTank(self.board, x, y, number)
            if mode == "A*":
                return QLearningTank(self.board, x, y, number, pretrained=True, save_file="qlearning_a_star.pkl")
            elif mode == "Planning-Graph":
                return QLearningTank(self.board, x, y, number, pretrained=True, save_file="qlearning_planning_graph.pkl.pkl")
            elif mode == "Minimax":
                return QLearningTank(self.board, x, y, number, pretrained=True, save_file="qlearning_minimax.pkl")
            elif mode == "Expectimax":
                return QLearningTank(self.board, x, y, number, pretrained=True, save_file="qlearning_expectimax.pkl")
            elif mode == "Random":
                return QLearningTank(self.board, x, y, number, pretrained=True, save_file="qlearning_random.pkl")

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
            self.board.update_mode(self.current_tank)
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
                valid_action = self.current_tank.shoot(direction_map[event.keysym])
            else:
                self.board.show_message("Invalid key!")

        if valid_action:
            self.board.update_bullet_count()  # Update bullet count after action
            self.board.update_bullets()  # Update bullet positions
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

        if self.turns >= MAX_TURNS:
            self.board.end_game("Draw!", winner=-1)

        # if isinstance(self.current_tank, AStarTank):
        #     self.board.delay_action(self.npc_act)
        # else:
        #     self.player_action_done = False
        if not isinstance(self.current_tank, PlayerTank):
            self.board.delay_action(self.npc_act)
        else:
            self.player_action_done = False

    def npc_act(self):
        """Perform action for NPC tank."""
        # check if the current tank is minimax or q-learning
        if isinstance(self.current_tank, MinimaxTank) or isinstance(self.current_tank, QLearningTank) or isinstance(
                self.current_tank, AStarTank) or isinstance(self.current_tank, ExpectimaxTank):
            self.current_tank.update()
        elif isinstance(self.current_tank, RandomTank):
            # choose randomly between moving and shooting - below 0.8 move, above 0.8 shoot
            import random
            choice = random.random()
            if choice < 0.8:
                self.current_tank.move(None)
            else:
                self.current_tank.shoot(None)
        else:
            self.current_tank.move(None)  # Move towards the target tank
            self.current_tank.shoot(None)  # Shoot if possible
        self.board.update_bullets()
        if not self.board.check_bullet_collisions():
            self.switch_turn()
