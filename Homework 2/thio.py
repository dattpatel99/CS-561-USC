from copy import deepcopy
import numpy as np
from go import GO

#####################################################################################
# Minimax Player Code. Uses minimax algorithm of depth 2 and several heuristics     #
# All credit goes to Me.                                                            #
#####################################################################################

class MiniMaxPLayer:
    def __init__(self):
        self.side = None

    def set_side(self, side):
        self.side = side

    def get_input(self, go, piece):
        self.set_side(piece)
        wall_breaker = self.detect_wall(go, piece_type)
        if wall_breaker:
            return wall_breaker
        if go.score(1) + go.score(2) < 10:
            if go.valid_place_check(2, 2, self.side):
                return [2, 2]
        possible_placements = self.get_possible_moves(go, self.side)
        if not possible_placements:
            answer = "PASS"
        else:
            answer = self.move(go)
            if answer == "":
                answer = possible_placements[0]
        return answer
    
    def detect_wall(self, go, piecet):
        # Need more space so dont wall
        if sum(p == piecet for p in go.board[1][1:4]) > 2:  # top
            if go.valid_place_check(2, 0, piecet):
                # capture opp inside fang on the mirror side
                if go.board[1][4] == 3 - piecet and go.board[2][4] == piecet and go.valid_place_check(0, 4, piecet):
                    return [0, 4]
                if go.board[3][0] == 3 - piecet and go.board[2][1] == 3 - piecet and go.valid_place_check(1, 0, piecet): # new
                    return [1, 0]
                # Dont Gang if it is surrounded
                if not (go.board[1][0] == 3 - piecet and go.board[2][1] == 3 - piecet):
                    return [2, 0]
                # Two Fangs
            if go.valid_place_check(2, 4, piecet):
                if go.board[1][0] == 3 - piecet and go.board[2][0] == piecet and go.valid_place_check(0, 0, piecet):
                    return [0, 0]

                if not (go.board[1][4] == 3 - piecet and go.board[2][3] == 3 - piecet):
                    return [2, 4]
        transpose = list(map(list, zip(*go.board)))
        if sum(p == piecet for p in transpose[1][1:4]) > 2:  # left
            if go.valid_place_check(0, 2, piecet):
                if go.board[4][1] == 3 - piecet and go.board[4][2] == piecet and go.valid_place_check(4, 3, piecet):
                    return [4, 3]
                if not (go.board[0][1] == 3 - piecet and go.board[1][2] == 3 - piecet):
                    return [0, 2]

            if go.valid_place_check(4, 2, piecet):
                if go.board[0][1] == 3 - piecet and go.board[0][2] == piecet and go.valid_place_check(0, 0, piecet):
                    return [0, 0]
                if not (go.board[4][1] == 3 - piecet and go.board[3][2] == 3 - piecet):
                    return [4, 2]

        # Have enough space so wall
        if sum(p == piecet for p in go.board[2][1:4]) > 2:  # middle
            if go.valid_place_check(2, 0, piecet):
                return [2, 0]
            if go.valid_place_check(2, 4, piecet):
                return [2, 4]

        if sum(p == piecet for p in transpose[2][1:4]) > 2:  # middle
            if go.valid_place_check(0, 2, piecet):
                return [0, 2]
            if go.valid_place_check(4, 2, piecet):
                return [4, 2]

        if sum(p == piecet for p in transpose[3][1:4]) > 2:  # right
            if go.valid_place_check(0, 2, piecet):
                if go.board[4][3] == 3 - piecet and go.board[4][2] == piecet and go.valid_place_check(4, 4, piecet):
                    return [4, 4]
                if not (go.board[0][3] == 3 - piecet and go.board[1][2] == 3 - piecet):
                    return [0, 2]

            if go.valid_place_check(4, 2, piecet):
                if go.board[0][3] == 3 - piecet and go.board[0][2] == piecet and go.valid_place_check(0, 4, piecet):
                    return [0, 4]
                if not (go.board[4][3] == 3 - piecet and go.board[3][2] == 3 - piecet):
                    return [4, 2]

        return None

    def get_possible_moves(self, go, piecetc):
        centers = []
        edge = []
        corner = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piecetc):
                    if (i == 0 and j == 0) or (i == 4 and j == 4) or (i == 0 and j == 4) or (i == 4 and j == 0):
                        corner.append((i, j))
                    elif i == 0 or i == 4 or j == 0 or j == 4:
                        edge.append((i, j))
                    else:
                        centers.append((i, j))
        return centers + edge + corner

    def move(self, go):          
        best_value, best_move, alpha, beta = self.minimax(go, True, 2, -np.inf, np.inf, None, None)
        return best_move

    def minimax(self, go, is_maximizing, depth, alpha, beta, branch, rm_pieces):
        best_move = ""

        if is_maximizing:
            best_value = -np.inf # assign alpha to -inf
            piece = self.side
        else:
            best_value = np.inf # assign alpha to inf
            piece = 3 - self.side # Switch to the other player if we are 1 then other is 3-1 while if we are 2 other is 3-2

        moves = self.get_possible_moves(go, piece) # Get set of moves
        opp_moves = self.get_possible_moves(go, 3 - piece) # get set of opp. moves

        if go.game_end(""):
            return [self.get_utility(go) * 5, "", alpha, beta]
        if depth == 0:
            return [self.get_utility_at_depth_new(go, rm_pieces), "", alpha, beta]
        if not moves:
            return [self.get_utility(go) * 0.5, "", alpha, beta]

        for move in moves:
            new_board = deepcopy(go)
            new_board.place_chess(move[0], move[1], piece)
            # Dont waste move on places white cant place unless its capture move
            rm_pc = new_board.remove_died_pieces(3 - piece)
            rm_pc_flag = False
            if len(rm_pc) == 0 and move not in opp_moves:
                rm_pc_flag = True
            backed_value = self.minimax(new_board, not is_maximizing, depth - 1, alpha, beta, branch, rm_pc_flag)[0]
            if is_maximizing is True and backed_value > best_value:
                best_value = backed_value
                best_move = move
                alpha = max(alpha, best_value)
            if is_maximizing is False and backed_value < best_value:
                best_value = backed_value
                best_move = move
                beta = min(beta, best_value)
            if alpha > beta:
                del new_board
                break
        return [best_value, best_move, alpha, beta]

    def get_utility(self, goboard):
        komi = -2.5 if self.side == 1 else 2.5
        return goboard.score(self.side) - goboard.score(3 - self.side) + komi
    
    def get_utility_at_depth_new(self, goboard, rm_pc):
        liberty_score = self.get_raw_liberty(goboard, self.side)
        connected_score = self.get_connected_count(goboard, self.side) * 2.5
        surrounding_score = self.get_surrounded_count(goboard, self.side)

        opp_liberty_score = self.get_raw_liberty(goboard, 3 - self.side) * 0.5
        opp_connected_score = self.get_connected_count(goboard, 3 - self.side)
        opp_surrounding_score = self.get_surrounded_count(goboard, 3 - self.side) * 0.8

        komi = -2.5 if self.side == 1 else 2.5
        score = (goboard.score(self.side) - goboard.score(3 - self.side) + komi) * 2

        f_core = liberty_score + connected_score + surrounding_score + (score) - \
               (opp_liberty_score) - opp_connected_score - (opp_surrounding_score)

        # if rm_pc:
        #     f_core = f_core * 0.5

        return f_core
    
    def get_raw_liberty(self, go, piece_loc):
        lib_count = 0
        for x in range(5):
            for y in range(5):
                if go.board[x][y] == piece_loc:
                    neighbors = go.detect_neighbor(x, y)
                    neigh_lib_count = 0
                    for piece in neighbors:
                        if go.board[piece[0]][piece[1]] == 0:
                            neigh_lib_count = neigh_lib_count + 1
                    if neigh_lib_count == 4:
                        lib_count = lib_count + 2.5
                    elif neigh_lib_count == 3:
                        lib_count = lib_count + 2.25
                    else:
                        lib_count = lib_count + (neigh_lib_count)
        return lib_count

    def get_connected_count(self, go, piece_loc):
        connected = [[False for x in range(5)] for y in range(5)]
        connected_count = 0
        for x in range(5):
            for y in range(5):
                if not connected[x][y] and go.board[x][y] == piece_loc:
                    allys = go.ally_dfs(x, y)

                    if len(allys) > 1:
                        for member in allys:
                            connected[member[0]][member[1]] = True
                        connected_count = connected_count + len(allys)
        return connected_count

    def get_surrounded_count(self, go, piece_loc):
        surrounded_count = 0
        for x in range(5):
            for y in range(5):
                if go.board[x][y] == (3 - piece_loc):
                    neighbors = go.detect_neighbor(x, y)
                    for piece in neighbors:
                        # Add to allies list if having the same color
                        if go.board[piece[0]][piece[1]] == piece_loc:
                            surrounded_count = surrounded_count + 1
        return surrounded_count * 0.8
   
# Credit to the TA Team
def read_input(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
        pt = int(lines[0])
        pb = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        b = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]
        return pt, pb, b

# Credit to the TA Team
def write_output(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])
    with open(path, 'w') as f:
        f.write(res)

if __name__ == "__main__":
    N = 5
    go_board = GO(5)
    piece_type, previous_board, board = read_input(N)
    go_board.set_board(piece_type, previous_board, board)
    player = MiniMaxPLayer()
    action = player.get_input(go_board, piece_type)
    write_output(action)