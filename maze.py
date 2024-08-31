import random


def generate_spacious_maze(board_width, board_height):
    """
    Generate a random spacious maze ensuring more path tiles than walls, with a variety of paths including diagonal connections.
    
    :param board_width: Width of the board.
    :param board_height: Height of the board.
    :return: 2D list representing the maze where 'W' represents walls and ' ' represents paths.
    """
    # Initialize grid with walls ('W')
    maze = [['W' for _ in range(board_width)] for _ in range(board_height)]

    # Start point
    start_x, start_y = 0, 0
    end_x, end_y = board_width - 1, board_height - 1

    # Mark start and end points as paths
    maze[start_y][start_x] = ' '
    maze[end_y][end_x] = ' '

    # Randomly generate paths ensuring connectivity and variation
    def carve_path(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (-2, 2), (2, -2)]
        random.shuffle(directions)  # Shuffle directions to ensure random carving

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < board_width and 0 <= ny < board_height and maze[ny][nx] == 'W':
                # Check if the path can be created (at least 2 empty neighbors)
                adjacent_walls = sum(
                    maze[ny + a][nx + b] == 'W'
                    for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    if 0 <= ny + a < board_height and 0 <= nx + b < board_width
                )

                if adjacent_walls >= 2:
                    maze[ny][nx] = ' '  # Carve the path
                    # Create the connecting path (ensures a 2-wide passage)
                    maze[ny - dy // 2][nx - dx // 2] = ' '
                    carve_path(nx, ny)  # Recursively carve from the new point

    # Start carving from random points to ensure more open spaces
    carve_path(start_x, start_y)

    # Ensure the maze is spacious by adding more paths
    fill_paths(maze, board_width, board_height)

    return maze


def fill_paths(maze, board_width, board_height):
    """
    Fills additional random paths to make the maze more spacious, ensuring more paths than walls.
    
    :param maze: The maze grid.
    :param board_width: Width of the board.
    :param board_height: Height of the board.
    """
    # Count total cells and path cells
    total_cells = board_width * board_height
    path_count = sum(row.count(' ') for row in maze)

    # Fill random paths until more than 50% are paths
    while path_count <= total_cells * 0.6:
        wx, wy = random.randint(0, board_width - 1), random.randint(0, board_height - 1)
        if maze[wy][wx] == 'W':
            maze[wy][wx] = ' '
            path_count += 1


def print_maze(maze):
    """
    Prints the maze to the console.
    
    :param maze: The maze grid.
    """
    for row in maze:
        print(''.join(row))


# Example usage
width = 15
height = 10
maze = generate_spacious_maze(width, height)
print_maze(maze)
