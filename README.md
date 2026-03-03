# monte-carlo-chess

Monte Carlo Tree Search (MCTS) implementation for chess using classical material evaluation.

This program uses Monte Carlo Tree Search combined with standard piece values (pawn=1, knight=3, bishop=3, rook=5, queen=9) to evaluate a given chess position and determine the best next move. The search simulates random playouts and estimates move quality based on visit counts and average evaluation.

## How to Run

```bash
pip install python-chess
python monte_carlo_sim.py
