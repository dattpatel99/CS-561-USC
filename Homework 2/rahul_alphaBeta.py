import rahul_host
import numpy as np
import random
import time
from copy import deepcopy
from read import *
from write import writeNextInput


class AlphaBetaPlayer():
    
    def __init__(self, piece_type, max_depth):
        self.type = 'alphabeta'
        self.piece_type = piece_type
        self.max_depth = max_depth
        
    def move(self, go, board, i, j, piece_type):
        #board = self.board

        '''valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)'''
        board[i][j] = piece_type
        go.board = board
        return board

    def moveLocations(self, go, board, piece_type):
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(board, i, j, piece_type, test_check = True):
                    n = go.detect_neighbor(board, i, j)
                    
                    eye = True
                    for nex in n:
                        if board[nex[0]][nex[1]] != piece_type:
                            eye = False
                    if not eye:
                        possible_placements.append((i,j))
                        
        return possible_placements

    def getTotalLiberty(self, go, piece_type):
        count = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    neighbors = go.detect_neighbor(go.board, i, j)
                    curr = 0
                    for k in neighbors:
                        if go.board[k[0]][k[1]] == 0:
                            curr += 1
                    count += curr
        return count

    def getSurround(self, go, piece_type):
        count = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == (3 - piece_type):
                    n = go.detect_neighbor(go.board, i, j)
                    for k in n:
                        if go.board[k[0]][k[1]] == piece_type:
                            count += 1
        return count

    def getConnected(self, go, piece_type):
        visited = [[False for i in range(go.size)]for j in range(go.size)]
        count = 0
        discount = 1
        for i in range(go.size):
            for j in range(go.size):
                if not visited[i][j] and go.board[i][j] == piece_type:
                    connections = go.ally_dfs(go.board, i, j)
                    for k in connections:
                        visited[k[0]][k[1]] = True
                    count += len(connections) * discount
                    discount *= 0.8
        return count

    def evaluationFunction(self, go, piece_type):
        if piece_type == 1:
            black = 1.5
            black2 = 1.5
        else:
            black = 1
            black2 = 1
        #Liberty
        s1 = self.getTotalLiberty(go, piece_type)
        s1e = self.getTotalLiberty(go, 3-piece_type)

        #Connected
        s2 = self.getConnected(go, piece_type)
        s2e = self.getConnected(go, 3-piece_type)

        #Surrounded
        s3 = self.getSurround(go, piece_type)
        s3e = self.getSurround(go, 3-piece_type)

        s4 = go.score(piece_type) - go.score(3-piece_type)
        
        '''if piece_type == 1:
            s3 = go.score(1) - (go.score(2)+go.komi)
        if piece_type == 2:
            s3 = (go.score(2)+go.komi) - go.score(1)'''
        
        return s4 + 0.8*(s1-s1e) + black*0.8*(s2-s2e) + black2*0.5*(s3-s3e)

    def minRec(self, go, alpha, beta, depth, piece_type, board, currtime):
        mini = np.inf
        possible_placements = self.moveLocations(go, board, piece_type)

        if len(possible_placements) == 0:
            return (-1, -1), self.evaluationFunction(go, self.piece_type)/2
        if depth == 0 or time.time() - currtime > 8:
            return (-1, -1), self.evaluationFunction(go, self.piece_type)

        for i in possible_placements:
            curr = deepcopy(go)
            new = self.move(curr, board, i[0], i[1], piece_type)
            curr.remove_died_pieces(3 - piece_type)
            move1, value = self.maxRec(curr, alpha, beta, depth-1, 3 - piece_type, new, currtime)
            if value < mini:
                mini = value
                best = i
            beta = min(value, beta)
            if beta <= alpha:
                break
        return best, mini
            

    def maxRec(self, go, alpha, beta, depth, piece_type, board, currtime):
        maxi = -np.inf
        possible_placements = self.moveLocations(go, board, piece_type)

                            
        if len(possible_placements) == 0:
            return (-1, -1), self.evaluationFunction(go, self.piece_type)/2
        if depth == 0 or time.time() - currtime > 8:
            return (-1, -1), self.evaluationFunction(go, self.piece_type)
        
        for i in possible_placements:
            curr = deepcopy(go)
            new = self.move(curr, curr.board, i[0], i[1], piece_type)
            curr.remove_died_pieces(3 - piece_type)
            move1, value = self.maxRec(curr, alpha, beta, depth-1, 3 - piece_type, new, currtime)
            if value > maxi:
                maxi = value
                best = i
            alpha = max(value, alpha)
            if beta <= alpha:
                break
        return best, maxi

    def choose_action(self, board, go, piece_type):
        random.seed(15)
        piece = piece_type
        possible_placements = self.moveLocations(go, board, piece_type)
        
        #Opening Moves
        if piece_type == 1:
            if len(possible_placements) > 15:
                if (2,2) in possible_placements:
                    return (2,2), 1
                openers = []
                openingMoves = [(1,2),(2,1),(2,3),(3,2)]
                for mov in openingMoves:
                    if mov in possible_placements:
                        openers.append(mov)
                if len(openers) > 0:
                    return random.choice(openers), 1
                '''openingMoves = [(1,1),(1,3),(3,1),(3,3),(2,0),(2,4),(0,2),(4,2)]
                for mov in openingMoves:
                    if mov in possible_placements:
                        openers.append(mov)
                if len(openers) > 0:
                    return random.choice(openers), 1'''

                
        #Aggresive Moves
        aggresivemoves = []
        for mov in possible_placements:
            self.move(go, board, mov[0], mov[1], piece_type)
            captures = go.find_died_pieces(go.board, 3 - piece_type)
            self.move(go, board, mov[0], mov[1], 0)
            if len(captures) > 0:
                aggresivemoves.append([mov,len(captures)])
            
        '''for mov in possible_placements:
            curr = deepcopy(go)
            self.move(curr, curr.board, mov[0], mov[1], piece_type)
            curr.remove_died_pieces(3 - piece_type)
            new = self.moveLocations(curr, curr.board, 3 - piece_type)
            for mov2 in new:
                self.move(curr, curr.board, mov2[0], mov2[1], 3-piece_type)
                captures = go.find_died_pieces(curr.board, piece_type)
                if len(captures) > 0:
                    possible_placements.remove(mov)
                    break'''
        
        #Prevent Captures
        prevention = []
        for mov in possible_placements:
            self.move(go, board, mov[0], mov[1], 3 - piece_type)
            captures = go.find_died_pieces(go.board, piece_type)
            self.move(go, board, mov[0], mov[1], 0)
            if len(captures) > 0:
                prevention.append([mov,len(captures)])
        
        if len(aggresivemoves) > 0:
            aggresivemoves.sort(key= lambda x:x[1], reverse=True)
            return tuple(aggresivemoves[0][0]), 1
        if len(prevention) > 0:
            prevention.sort(key= lambda x:x[1], reverse=True)
            return tuple(prevention[0][0]), 1
                
        action, score = self.maxRec(go, -np.inf, np.inf, self.max_depth, piece_type, board, time.time())
        return action, score
