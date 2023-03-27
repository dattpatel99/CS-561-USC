import copy
from host import GO
import numpy as np
import random
from time import time

random.seed(10)
class MinMax:
    def __init__(self):
        self.side= None
        self.depth = 3
        self.lib_weight= 0
        self.opp_lib_weight= 0
        self.connect_weight= 0
        self.opp_connect_weight= 0
        self.score_weight = 0
        self.edgeLocations = [[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]]
        self.play_as_black = 0
    
    def set_side(self, side):
        self.side = side

    def update_heuristic_constants(self, go):
        self.play_as_black = 2.8 if self.side == 1 else 1
        tot_pieces = go.n_move
        if(0<=tot_pieces<10):
            self.connect_weight = 1.9
            self.opp_connect_weight = 0.4 * self.play_as_black
            self.lib_weight= 1.2
            self.opp_lib_weight= 0.8
            self.score_weight = 1.2
        elif(10<=tot_pieces<17):
            self.connect_weight = 1.5
            if(14<=tot_pieces):
                self.opp_connect_weight = 0.9 * self.play_as_black
                self.opp_lib_weight= 0.4
            self.lib_weight= 1.9 * self.play_as_black
            self.score_weight = 1.9
        elif(17<=tot_pieces<22):
            self.connect_weight = 1.5
            self.opp_connect_weight = 0.8 * self.play_as_black
            self.lib_weight= 1 * self.play_as_black
            self.opp_lib_weight= 1.5
            self.score_weight = 2.8
        else:
            self.connect_weight = 0.4
            self.opp_connect_weight = 0.7
            self.lib_weight= 1
            self.opp_lib_weight= 1.9
            self.score_weight = 4

    def all_heuristics(self, go, end):
        score = self.evalutaion(go) * 2
        if end:
            return score
        side = self.side
        
        pieces = {}
        opp_pieces = {}
        empty_checked = {}
        
        for i in range(5):
            for k in range(5):
                if(go.board[i][k] == 0):
                    empty_checked[(i,k)] = 0
                elif(go.board[i][k] == side):
                    pieces[(i,k)] = 0
                else:
                    opp_pieces[(i,k)] = 0
        
        # Liberty for a piece
        lib = self.get_liberities(go, pieces) * self.lib_weight
        connects = self.connected_pieces(go, copy.deepcopy(pieces)) * 1.5 *  self.connect_weight

        # Opp Liberty
        opp_lib = self.get_liberities(go, opp_pieces) * 0.6 * self.opp_lib_weight
        opp_connects = self.connected_pieces(go, copy.deepcopy(opp_pieces)) * 0.7 * self.opp_connect_weight
        
        # Score
        state_score = (lib+connects)-(opp_lib+opp_connects) + score

        return state_score

    # Use this based on piece types
    def connected_pieces(self, go, pieces):
        connects = 0
        for pos in list(pieces.keys()):
            pieces[pos] = 1
            allies = go.ally_dfs(pos[0], pos[1])
            if(len(allies)>1):
                connects += len(allies)
            for ally in allies:
                pieces[ally] = 1
        return connects

    # HEURISTICS
    def get_liberities(self, go, pieces):
        empty_checked = {}
        for i in range(5):
            for k in range(5):
                if(go.board[i][k] == 0):
                    empty_checked[(i,k)] = 0
        libteries = 0
        for pos in list(pieces.keys()):
            neigh = go.detect_neighbor(pos[0],pos[1])
            for n in neigh:
                if(go.board[n[0]][n[1]] == 0 and empty_checked[(n[0],n[1])] == 0):
                    libteries += 1
                    empty_checked[(n[0],n[1])] = 1
        return libteries

    # UTILITY FUNCTIONS
    def evalutaion(self, go):
        komi = -go.komi if self.side == 1 else go.komi # Changing komi based on white or black
        return go.score(self.side) - go.score(3-self.side) + komi

    # Get empty positions based on board
    # Single loop
    def possible_pos(self, go, piece):
        empty_locs = []
        for i in range(go.size):
            for j in range(go.size):
                val = go.board[i][j]
                if(val == 0 and go.valid_place_check(i,j, piece)):
                    # If it is less than 10 moves then only choose non-edge locations
                    if(go.n_move < 10 and self.edgeLocations[i][j] == 0):
                        empty_locs.append((i,j))
                    else:
                        # If moves is greater than 18 then 
                        if(go.n_move > 18):
                            # Check for allies. If the number of alies is less than 2 then only take that as a valid location.
                            n = go.detect_neighbor(i,j)
                            a = 0
                            for each in n:
                                if(go.board[each[0]][each[1]] == piece):
                                    a += 1
                            if (a <= 2):
                                empty_locs.append((i,j))
                        else:
                            empty_locs.append((i,j))
        return empty_locs

    # MINI-MAX
    def mini_max(self, maximize, depth, alpha, beta, piece, go):
        best_move = ()

        # Indicate whether we are maximizing or minimizing
        if(maximize):
            best_val = float("-inf")
        else:
            best_val = float('inf')

        # Get list of possible moves
        moves = self.possible_pos(go, piece)
        # If this is a terminal state
        if go.game_end(''):
            return [self.all_heuristics(go, True), "", alpha, beta]
        # If the prev board == current board and current moves player is winning then choose PASS again and end the game
        if (go.compare_board(go.previous_board, go.board) and go.judge_winner() == piece):
            return [self.all_heuristics(go,True), "PASS", alpha, beta]
        # There could be a terminiation move but depth reached
        if depth == 0:
            return [self.all_heuristics(go, False) * 0.2, "", alpha, beta]
        # Worst there is no move
        if not moves:
            return [self.all_heuristics(go, False) * 0.01, "", alpha, beta]

        # Minimax loop
        moves.append("PASS")
        random.shuffle(moves)

        for possible in moves:
            # Make Move
            temp_go = copy.deepcopy(go)
            if possible != "PASS":                
                temp_go.place_chess(possible[0],possible[1],piece)
                temp_go.remove_died_pieces(3-piece)
            else:
                # If the move is a PASS then our new go object has to update the following: n_move
                temp_go.previous_board = temp_go.board
                temp_go.n_move += 1
            self.update_heuristic_constants(temp_go)
            result = self.mini_max(not maximize, depth-1, alpha, beta, 3-piece, temp_go)
            # Update Alpha and beta value
            if maximize and result[0] > best_val:
                best_val = result[0]
                best_move = possible
                alpha = max(alpha, best_val)
            if not maximize and result[0] < best_val:
                best_val = result[0]
                best_move = possible
                beta = min(beta, best_val)
            if alpha > beta:
                del temp_go
                break
        return [best_val, best_move, alpha, beta]
    
    def get_attack_list(self, empty_locs, go, side):
        temp = {}
        for pos in empty_locs:
            # DIRECT OFFENSE Check for moves where I kill opp
            # Check number of oppo. move kills
            go.board[pos[0]][pos[1]] = self.side
            opp_dead = go.find_died_pieces(3-self.side)
            go.board[pos[0]][pos[1]] = 0
            if len(opp_dead)>0:
                temp[pos] = len(opp_dead)
        return temp

    def get_move(self, go):
        empty_locs = self.possible_pos(go, self.side)
        random.shuffle(empty_locs)

        # Attacking moves:
        my_attack_moves = self.get_attack_list(empty_locs, go, self.side)
        opp_attack_moves = self.get_attack_list(empty_locs, go, 3-self.side)

        # Best move based on kills/saved
        best_move = {}
        for move in list(my_attack_moves.keys()):
            if move in list(opp_attack_moves.keys()):
                best_move[move] = my_attack_moves[move]/opp_attack_moves[move]
                del my_attack_moves[move]
                del opp_attack_moves[move]

        # Sort by most killed opponents
        my_attack_moves = {k: v for k, v in sorted(my_attack_moves.items(), key=lambda item: item[1])}
        opp_attack_moves = {k: v for k, v in sorted(opp_attack_moves.items(), key=lambda item: item[1])}
        best_move = {k: v for k, v in sorted(best_move.items(), key=lambda item: item[1])}

        # Play smart
        if(len(best_move) != 0):
            return list(best_move.keys())[0]
        # Play offensive
        elif(len(my_attack_moves) != 0):
            return list(my_attack_moves.keys())[0]
        # Play defensive
        elif(len(opp_attack_moves) != 0):
            return list(opp_attack_moves.keys())[0]
        if(not empty_locs):
            return "PASS"
        else:
            # Last option: Run minimax
            result = self.mini_max(True, self.depth, -np.inf, np.inf, self.side, go)
            if result[1] == "":
                random.shuffle(empty_locs)
                return empty_locs[0]
            else:
                return result[1]

# Credit to TA team 
def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
        piece_type = int(lines[0])
        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]
        return piece_type, previous_board, board

# Credit to TA team 
def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')
        if position[0] == "PASS":
            return "PASS", -1, -1
        x = int(position[0])
        y = int(position[1])
    return "MOVE", x, y

# Credit to TA team 
def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])
    with open(path, 'w') as f:
        f.write(res)

# Credit to TA team 
def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")

# Credit to TA team 
def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"
    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"
    with open(path, 'w') as f:
        f.write(res[:-1])

if __name__ == "__main__":
    N = 5
    go_board = GO(5)
    piece_type, previous_board, board = readInput(N)
    go_board.set_board(piece_type, previous_board, board)

    # If the previous move is a PASS by opp and we are currently winning then end the game with a PASS
    if(go_board.compare_board(previous_board,board) and go_board.judge_winner() == piece_type):
        action = "PASS"
    else:
        # Estimate total number of moves
        num_pieces = 0
        for r in board:
            for c in r:
                if(c != 0):
                    num_pieces+=1
        go_board.n_move = num_pieces        
        player = MinMax()
        player.update_heuristic_constants(go_board)
        player.set_side(piece_type)
        action = player.get_move(go_board)
    print(action)
    writeOutput(action)