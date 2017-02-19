"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

# Constants for extreme cases
NEGATIVE_INFINITY = float("-inf")
INFINITY = float("inf")


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def score_two_moves_ahead(game, player, factor):
    """
    Based on isolation.get_legal_moves() this function figures out all of the open squares that a player could possibly
    move to within the next 2 moves (ignoring opponent's moves) and computes a score based on a weighted sum of the
    number of results from one and two moves ahead.
    :param game: `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    :param player:  object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    :param factor: float
        The final score will be m1 + (factor * m2) where m1 = number of spots availabe to move to in 1 move and
        m2 = number of spots available to move to in 2 moves
    :return: float
        The weighted sum of the numver of spots available in 1 move and in 2 moves.
    """
    r, c = game.get_player_location(player)

    first_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2), (1, 2), (2, -1), (2, 1)]
    second_moves = [(-4, -2), (-4, 0), (-4, 2),
                    (-3, -3), (-3, -1), (-3, 1), (-3, 3),
                    (-2, -4), (-2, 0), (-2, 4),
                    (-1, -3), (-1, -1), (-1, 1), (-1, 3),
                    (0, -4), (0, -2), (0, 2), (0, 4),
                    (1, -3), (1, -1), (1, 1), (1, 3),
                    (2, -4), (2, 0), (2, 4),
                    (3, -3), (3, -1), (3, 1), (3, 3),
                    (4, -2), (4, 0), (4, 2)]

    valid_first_moves = [(r + dr, c + dc) for dr, dc in first_moves if game.move_is_legal((r + dr, c + dc))]
    valid_second_moves = [(r + dr, c + dc) for dr, dc in second_moves if game.move_is_legal((r + dr, c + dc))]

    return len(valid_first_moves) + len(valid_second_moves) * factor


def heuristic_a(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Heuristic A - Computes all possible moves for 1 and 2 moves ahead for each player and computes the difference.
    # A weighting factor is applied to future moves to represent the uncertainty of the unknown outcome of the first
    # move. Leaves with winning or losing positions are scored +inf and -inf respectively.
    if game.is_loser(player):
        return NEGATIVE_INFINITY

    if game.is_winner(player):
        return INFINITY

    player_score = score_two_moves_ahead(game, player, 0.9)
    opponent_score = score_two_moves_ahead(game, game.get_opponent(player), 0.8)

    return player_score - opponent_score


def heuristic_b(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Heuristic B - Compute the difference between the number of legal moves available to the players. Increase
    # the score by the count of every opponenet's move that might be occupied by a player's move. Leaves with winning or
    # losing positions are scored +inf and -inf respectively.
    if game.is_loser(player):
        return NEGATIVE_INFINITY

    if game.is_winner(player):
        return INFINITY

    player_moves = game.get_legal_moves(player)
    num_player_moves = len(player_moves)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    num_openent_moves = len(opponent_moves)
    num_same_moves = len(set(player_moves).intersection(set(opponent_moves)))

    return float(num_player_moves - num_openent_moves + num_same_moves)


def heuristic_c(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Heuristic C - Compute the weighted difference between the number of legal moves available to the players.
    #  Leaves with winning or losing positions are scored +inf and -inf respectively.
    if game.is_loser(player):
        return NEGATIVE_INFINITY

    if game.is_winner(player):
        return INFINITY

    num_player_moves = len(game.get_legal_moves(player))
    num_opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return num_player_moves - 0.5 * num_opponent_moves


# Selects the heuristic to use
custom_score = heuristic_a


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if len(legal_moves) == 0:
            return -1, -1

        # If we time out, at least we can pick a move
        best_move = legal_moves[0]

        # Determine which method to call (keeping this out of the inner loop)
        method = self.minimax if self.method == 'minimax' else self.alphabeta

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            depth = 1
            if self.iterative:
                while True:
                    _, best_move = method(game, depth)
                    depth += 1
            else:
                _, best_move = method(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # raise NotImplementedError
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1, -1)

        player = game.active_player if maximizing_player else game.inactive_player

        if depth == 0:
            # This is as deep as we'll go
            return self.score(game, player), best_move

        moves = game.get_legal_moves()
        if len(moves) == 0:
            # There are no moves left.
            return self.score(game, player), best_move

        if maximizing_player:
            best_score = NEGATIVE_INFINITY
            for move in moves:
                forecast = game.forecast_move(move)
                forecast_score, _ = self.minimax(forecast, depth - 1, False)
                if forecast_score > best_score:
                    best_score = forecast_score
                    best_move = move
        else:
            best_score = INFINITY
            for move in moves:
                forecast = game.forecast_move(move)
                forecast_score, _ = self.minimax(forecast, depth - 1, True)
                if forecast_score < best_score:
                    best_score = forecast_score
                    best_move = move

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move = (-1, -1)

        player = game.active_player if maximizing_player else game.inactive_player

        if depth == 0:
            return self.score(game, player), best_move

        moves = game.get_legal_moves()
        if len(moves) == 0:
            return self.score(game, player), best_move

        if maximizing_player:
            best_score = NEGATIVE_INFINITY
            for move in moves:
                forecast = game.forecast_move(move)
                forecast_score, _ = self.alphabeta(forecast, depth - 1, alpha, beta, False)
                if forecast_score > best_score:
                    best_score = forecast_score
                    best_move = move
                    if best_score > alpha:
                        alpha = best_score
                        if beta <= alpha:
                            break
        else:
            best_score = INFINITY
            for move in moves:
                forecast = game.forecast_move(move)
                forecast_score, _ = self.alphabeta(forecast, depth - 1, alpha, beta, True)
                if forecast_score < best_score:
                    best_score = forecast_score
                    best_move = move
                    if best_score < beta:
                        beta = best_score
                        if beta <= alpha:
                            break

        return best_score, best_move
