from rahul_host import GO
import numpy as np
from read import readInput
from write import writeOutput
from write import writeNextInput
from rahul_alphaBeta import AlphaBetaPlayer

if __name__ == "__main__":
    N = 5
    go = GO(N)
    piece_type, previous_board, board = readInput(N)
    depth = 4
    player = AlphaBetaPlayer(piece_type, depth)
    go.set_board(piece_type, previous_board, board)
    action, score = player.choose_action(go.board, go, piece_type)
    if action == (-1,-1):
        action = "PASS"
    print(action)
    writeOutput(action)
