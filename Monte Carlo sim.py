import math
import random
import chess

# Classical evaluation of pieces
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def evaluate_material(board):
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def terminal_value(board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    return None


def rollout(board, depth=20):
    term = terminal_value(board)
    if term is not None:
        return term

    for _ in range(depth):
        term = terminal_value(board)
        if term is not None:
            return term
        moves = list(board.legal_moves)
        if not moves:
            return 0.0
        board.push(random.choice(moves))

    score = evaluate_material(board)
    return math.tanh(score / 600.0)


class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = list(board.legal_moves)
        self.visits = 0
        self.value_sum = 0.0

    def uct(self, child, c=1.4):
        if child.visits == 0:
            return float("inf")
        exploitation = child.value_sum / child.visits
        exploration = c * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration

    def select_child(self):
        return max(self.children, key=lambda c: self.uct(c))

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self, move=move)
        self.children.append(child)
        return child

    def backprop(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backprop(value)


def mcts(board, iterations=5000):
    root = Node(board.copy())

    for _ in range(iterations):
        node = root

        # Part 1: Selection
        while not node.untried_moves and node.children:
            node = node.select_child()

        # Part 2: Expansion
        if node.untried_moves:
            node = node.expand()

        # Part 3: Simulation
        sim_board = node.board.copy()
        value = rollout(sim_board)

        # Backpropagation
        node.backprop(value)

    # Best move = most visits
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child



# Test
if __name__ == "__main__":
    random.seed(0)

    fen = "8/3q4/2pppk2/8/2PPPK2/3Q4/8/8 w - - 0 1"
    board = chess.Board(fen)

    print("Starting position:")
    print(board)
    print("\nRunning MCTS...\n")

    best = mcts(board, iterations=8000)

    print("Best move:", best.move)
    print("Visits:", best.visits)
    print("Estimated value (white):", best.value_sum / best.visits)