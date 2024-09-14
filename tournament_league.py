import visualizations
from visualizations import *

BOARD_WIDTH, BOARD_HEIGHT = 10, 10  # Board dimensions


class ResultsTracker:
    def __init__(self, tournament):
        self.tank1_wins = 0
        self.tank2_wins = 0
        self.draws = 0
        self.tournament = tournament

    def add_tank1_win(self):
        self.tank1_wins += 1
        self.tournament.visualize_results()
        # self.tournament.next_game()
        self.tournament.results_update()

    def add_tank2_win(self):
        self.tank2_wins += 1
        self.tournament.visualize_results()
        # self.tournament.next_game()
        self.tournament.results_update()

    def add_draw(self):
        self.draws += 1
        self.tournament.visualize_results()
        # self.tournament.next_game()
        self.tournament.results_update()


class NonVisualBoard:
    def __init__(self, width, height, main_window, results_tracker):
        """
        Initialize the game board.

        :param width: Width of the board.
        :param height: Height of the board.
        :param main_window: The main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        """
        self.results_tracker = results_tracker

        self.width = width  # Width of the board
        self.height = height  # Height of the board

        self.grid = [[GameConstants.BOARD for _ in range(width)] for _ in range(height)]  # Grid of colors
        self.generate_maze()

        self.main_window = main_window  # Main window reference

        self.tank1 = None  # Reference to tank 1
        self.tank2 = None  # Reference to tank 2
        self.bullets = []  # List of bullets
        self.mode = 'move'  # Current mode (move or shoot)
        self.turns = 0  # Turn counter

        self.ended = False

    def draw_grid(self):
        """Draw the grid on the canvas"""
        # clear the console
        print("\033[H\033[J")  # if doesn't work, try printing 100 new lines
        # print the grid
        for y in range(self.height):
            for x in range(self.width):
                print(self.grid[y][x], end=' ')
            print()

    def place_tank(self, tank, number):
        """
        Place a tank on the board.

        :param tank: Tank object to place.
        :param number: Tank number (1 or 2).
        """
        if number == 1:
            self.tank1 = tank
            color = GameConstants.TANK1
        else:
            self.tank2 = tank
            color = GameConstants.TANK2
        self.update_position(tank.x, tank.y, color)

    def update_position(self, x, y, color):
        """
        Update the position of an object on the board.

        :param x: X coordinate.
        :param y: Y coordinate.
        :param color: Color of the object.
        """
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
            color = GameConstants.TANK1 if number == 1 else GameConstants.TANK2
            self.update_position(tank.x, tank.y, GameConstants.BOARD)  # Repaint old position
            tank.x, tank.y = new_x, new_y
            self.update_position(new_x, new_y, color)
            return True
        else:
            # self.show_message("Invalid move!")
            return False

    def add_bullet(self, bullet):
        """
        Add a bullet to the board.

        :param bullet: Bullet object to add.
        """
        if self.is_valid_move(bullet.x, bullet.y):
            self.bullets.append(bullet)
            self.update_position(bullet.x, bullet.y, GameConstants.BULLET)
            return True
        return False

    def move_bullet(self, bullet, new_x, new_y):
        """
        Move a bullet to a new position.

        :param bullet: Bullet object to move.
        :param new_x: New X coordinate.
        :param new_y: New Y coordinate.
        """
        self.update_position(bullet.x, bullet.y, GameConstants.BOARD)  # Repaint old position
        bullet.x, bullet.y = new_x, new_y
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.update_position(new_x, new_y, GameConstants.BULLET)

    def remove_bullet(self, bullet):
        """
        Remove a bullet from the board.

        :param bullet: Bullet object to remove.
        """
        self.update_position(bullet.x, bullet.y, GameConstants.BOARD)  # Repaint old position with brown
        if bullet in self.bullets:
            self.bullets.remove(bullet)

    def check_bullet_collisions(self):
        """
        Check for bullet collisions with tanks.

        :return: True if a collision occurs, False otherwise.
        """
        for bullet in self.bullets:
            if bullet.x == self.tank1.x and bullet.y == self.tank1.y:
                self.end_game(f"Tank 2 ({self.tank2.__class__.__name__}) wins!", 2)
                return True
            elif bullet.x == self.tank2.x and bullet.y == self.tank2.y:
                self.end_game(f"Tank 1 ({self.tank1.__class__.__name__}) wins!", 1)
                return True
        return False

    def is_valid_move(self, x, y):
        """
        Check if a move is valid.

        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the move is valid, False otherwise.
        """
        return 0 <= x < self.width and 0 <= y < self.height and (self.grid[y][x] == GameConstants.BOARD or
                                                                 self.grid[y][x] == GameConstants.TANK1 or
                                                                 self.grid[y][x] == GameConstants.TANK2)

    def show_message(self, message):
        """
        Display a message.

        :param message: Message to display.
        """
        print(message)

    def update_mode(self, current_tank):
        """Update the mode label based on the current tank's type."""
        # instead of writing tank 1 and tank 2, we can write their types

        if isinstance(self.tank1, PlayerTank):
            print(f"Mode: {self.mode.capitalize()}")

        if isinstance(self.tank2, PlayerTank):
            print(f"Mode: {self.mode.capitalize()}")

    def update_bullets(self):
        """Update the positions of all bullets."""
        for bullet in self.bullets:
            if bullet.moves > 0:
                bullet.move()
            else:
                bullet.moves += 1
            if bullet.moves >= 10:
                self.remove_bullet(bullet)

    def end_game(self, result_message, winner):
        """
        End the game and show the result.

        :param result_message: Message to display.
        """
        # print(result_message)
        self.ended = True
        if winner == 1:
            self.results_tracker.add_tank1_win()
        elif winner == 2:
            self.results_tracker.add_tank2_win()
        else:
            self.results_tracker.add_draw()

    def quit_game(self):
        """Quit the game."""
        self.main_window.quit()

    def delay_action(self, func):
        """
        Execute an action with an optional delay.

        :param func: Function to execute.
        """
        func()

    def generate_maze(self):
        """Generate the maze and update the grid."""
        maze = generate_spacious_maze(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                if maze[y][x] == 'W':
                    self.grid[y][x] = GameConstants.WALL  # Use wall color for walls
                else:
                    self.grid[y][x] = GameConstants.BOARD  # Use board color for paths

    def is_wall(self, x, y):
        """
        Check if a position is a wall.

        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the position is a wall, False otherwise.
        """
        return x < 0 or x >= self.width or y < 0 or y >= self.height or self.grid[y][x] == GameConstants.WALL
    
    def is_tank(self, x, y):
        """
        Check if a position has a tank.

        :param x: X coordinate.
        :param y: Y coordinate.
        :return: True if the position has a tank, False otherwise.
        """
        return (self.tank1.x == x and self.tank1.y == y) or (self.tank2.x == x and self.tank2.y == y)

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


class NonVisualGame:
    def __init__(self, tank1_type, tank2_type, main_window, results_tracker, qmode1="None", qmode2="None"):
        """
        Initialize the game.

        :param main_window: Reference to the main Tkinter window.
        :param delay: Boolean to enable/disable delay for NPC actions.
        :param tank1_type: Type of tank 1
        :param tank2_type: Type of tank 2
        """
        self.results_tracker = results_tracker
        self.board = NonVisualBoard(BOARD_WIDTH, BOARD_HEIGHT, main_window, results_tracker)  # Game board
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
                return QLearningTank(self.board, x, y, number, pretrained=True,
                                     save_file="qlearning_planning_graph.pkl")
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
            # self.board.update_bullet_count()  # Update bullet count after action
            self.board.update_bullets()  # Update bullet positions
            self.player_action_done = True
            self.last_keys.clear()
            if not self.board.check_bullet_collisions():
                self.switch_turn()

    def switch_turn(self):
        """Switch turns between tanks."""
        self.turns += 1
        if self.turns >= GameConstants.MAX_TURNS:
            self.board.end_game("Game ended in a draw!", -1)
            return

        if self.current_tank == self.tank1:
            self.current_tank = self.tank2
        else:
            self.current_tank = self.tank1

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
            self.current_tank.act()
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


class TournamentLeague:
    def __init__(self, tank1_type, tank2_type, main_window, qmode1="None", qmode2="None", num_games=10,
                 amount_of_visualizations=0, popups=True):
        self.results_tracker = ResultsTracker(self)
        self.popups = popups
        self.num_games = num_games
        self.main_window = main_window
        self.tank1_type = tank1_type
        self.tank2_type = tank2_type
        self.qmode1 = qmode1
        self.qmode2 = qmode2

        self.results = []
        self.current_game = 0

        self.visualize_game_count = amount_of_visualizations

        # create an updating results window
        if self.popups:
            self.results_window = tk.Toplevel(self.main_window)
            self.results_window.title("Tournament Results")

            # create a label to display the results
            self.results_label = tk.Label(self.results_window, text="Tournament Results")
            self.results_label.pack()

            self.tank1_wins_label = tk.Label(self.results_window,
                                             text=f"Tank 1 ({self.tank1_type}) wins: {self.results_tracker.tank1_wins}")
            self.tank1_wins_label.pack()

            self.tank2_wins_label = tk.Label(self.results_window,
                                             text=f"Tank 2 ({self.tank2_type}) wins: {self.results_tracker.tank2_wins}")
            self.tank2_wins_label.pack()

            self.draws_label = tk.Label(self.results_window, text=f"Draws: {self.results_tracker.draws}")
            self.draws_label.pack()

            self.current_game_label = tk.Label(self.results_window, text=f"Current Game: {self.current_game}")
            self.current_game_label.pack()

        self.begin_tournament()

    def begin_tournament(self):
        visualized_games = []
        while self.current_game < self.num_games:
            self.current_game += 1
            try:
                if self.visualize_game_count <= 0:
                    NonVisualGame(self.tank1_type, self.tank2_type, self.main_window, self.results_tracker,
                                  qmode1=self.qmode1, qmode2=self.qmode2)
                else:
                    self.visualize_game_count -= 1
                    g = visualizations.Game(self.main_window, True, self.tank1_type, self.tank2_type,
                                            result_tracker=self.results_tracker, enable_endscreen=False,
                                            qmode1=self.qmode1,
                                            qmode2=self.qmode2)
                    visualized_games.append(g)
            except Exception as e:
                # if there was an error, count it as a draw
                print(f"Error in game {self.current_game}: {e}")
                self.results_tracker.add_draw()
            self.visualize_results()

    def results_update(self):
        if self.results_tracker.tank1_wins + self.results_tracker.tank2_wins + self.results_tracker.draws >= self.num_games:
            self.end_tournament()

    # def next_game(self):
    #     self.current_game += 1
    #     if self.current_game < self.num_games:
    #         if self.visualize_game_count <= 0:
    #             NonVisualGame(self.tank1_type, self.tank2_type, self.main_window, self.results_tracker, qmode1=self.qmode1, qmode2=self.qmode2)
    #         else:
    #             self.visualize_game_count -= 1
    #             visualizations.Game(self.main_window, True, self.tank1_type, self.tank2_type, result_tracker=self.results_tracker, enable_endscreen=False, qmode1=self.qmode1, qmode2=self.qmode2)
    #     self.visualize_results()

    def visualize_results(self):
        if not self.popups:
            return
        self.tank1_wins_label.config(text=f"Tank 1 ({self.tank1_type}) wins: {self.results_tracker.tank1_wins}")
        self.tank2_wins_label.config(text=f"Tank 2 ({self.tank2_type}) wins: {self.results_tracker.tank2_wins}")
        self.draws_label.config(text=f"Draws: {self.results_tracker.draws}")
        self.current_game_label.config(text=f"Current Game: {self.current_game}")

    def end_tournament(self):
        print(f"Tank 1 ({self.tank1_type}) wins: {self.results_tracker.tank1_wins}")
        print(f"Tank 2 ({self.tank2_type}) wins: {self.results_tracker.tank2_wins}")
        print(f"Draws: {self.results_tracker.draws}")

        if self.popups:
            visualizations.EndScreen(self.main_window, "end tournament")


class MegaTournament:  # does tournaments between all duos of tanks
    def __init__(self, tank_types, main_window, num_games=10, amount_of_visualizations=0):
        self.tank_types = tank_types
        self.main_window = main_window
        self.num_games = num_games
        self.visualize_game_count = amount_of_visualizations

        self.tournament_results = {}
        self.current_tournament = 0
        self.tournament_pairs = []

        for i in range(len(self.tank_types)):
            for j in range(len(self.tank_types)):  # range(i + 1, len(self.tank_types)):
                self.tournament_pairs.append((self.tank_types[i], self.tank_types[j]))

        self.begin_mega_tournament()

    def begin_mega_tournament(self):
        while self.current_tournament < len(self.tournament_pairs):
            tank1_type, tank2_type = self.tournament_pairs[self.current_tournament]
            final_tank1_type, final_tank2_type = tank1_type, tank2_type
            qmode1, qmode2 = "None", "None"
            if "Q-Learning" in tank1_type:
                qmode1 = tank1_type.split("|")[1]
                final_tank1_type = "Q-Learning"
            if "Q-Learning" in tank2_type:
                qmode2 = tank2_type.split("|")[1]
                final_tank2_type = "Q-Learning"
            self.current_tournament += 1
            print(f"Tournament {self.current_tournament} - {tank1_type} vs {tank2_type}")
            try:
                TournamentLeague(final_tank1_type, final_tank2_type, self.main_window, num_games=self.num_games,
                                 amount_of_visualizations=self.visualize_game_count,
                                 popups=False, qmode1=qmode1, qmode2=qmode2)
            except Exception as e:
                print(f"Error in tournament {self.current_tournament}: {e}")
                continue

        self.end_mega_tournament()

    def end_mega_tournament(self):
        print("Mega Tournament Ended")
        print(self.tournament_results)
        self.main_window.pack_forget()
        visualizations.EndScreen(self.main_window, "end mega tournament")
