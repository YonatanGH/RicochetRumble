INCLUDED FILES:
1. main.py - main file, RUN THIS
2. tanks.py - contains all of the tank classes
3. visualizations.py - manages the visualizations of the menus and game
4. tournament_league.py - manages the tournaments' logic
5. game_constants.py - contains all the game constants
6. maze.py - creates mazes for the game
7. bullet.py - contains the bullet class
UTIL FILES used by specific algorithms for the tanks:
8. action.py - used by the planning tank
9. action_layer.py - used by the planning tank
10. util.py - contains some utils
11. search.py - used by the planning tank
12. graph_plan.py - used by the planning tank
13. pgparser.py - used by the planning tank
14. plan_graph_level.py - used by the planning tank
15. proposition.py - propositions for the planning algorithm tank
16. proposition_layer.py - used by the planning tank
17. planning_problem.py - used by the planning tank
18. plan_domain.txt - a dummy domain file for the planning problems (changes in game)
19. plan_problem.txt - a dummy problem file for the planning problems (changes in game)
20. qlearning_a_star.pkl - a pickle file containing the Q-table for the Q-learning tank that was trained on the A* tank
21. qlearning_random.pkl - a pickle file containing the Q-table for the Q-learning tank that was trained on the random tank
22. qlearning_planning_graph.pkl - a pickle file containing the Q-table for the Q-learning tank that was trained on the planning graph tank
23. qlearning_minimax.pkl - a pickle file containing the Q-table for the Q-learning tank that was trained on the minimax tank
24. qlearning_expectimax.pkl - a pickle file containing the Q-table for the Q-learning tank that was trained on the expectimax tank

~~~~~~~~~ HOW TO RUN ~~~~~~~~~
1. Run main.py
2. For a quick game: choose tank1 and tank2, and press 'Start Game'
3. If you want to use a Q-learning tank, make sure to go to "Q-Settings" and choose the Q-table you want to use
4. For a tournament, press "Tournaments", and then choose the tanks you want to participate in the tournament
   and also choose the number of game you want the tanks to play against each other
   You can also choose how many of the games you want to visualize.
   Then press "Start Tournament".

USAGE OF LLMS:
we have used chatGPT and github copilot to help us with the code.
We have also used code from our previous exercises in the course.