from copy import deepcopy
from random import choice, choices
from math import sqrt, log
from tqdm import tqdm


class MCTS_Node:
    
    def __init__(self, parent, last_player, state, prior):
        self.parent = parent
        self.player = 1 - last_player
        self.state = state
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior
        self.UCB = float('inf')
        self.children = []
        self.prob_dist = None
    
    def is_leaf(self):
        # if there is no children
        if len(self.children) == 0:
            return True
        else:
            return False
    
    def is_terminal(self):
        # if the game is over
        if self.state.is_game_over():
            return True
        else:
            return False
    
    def create_children(self):
        # create a new node for every possible next game state
        legal_moves = list(self.state.legal_moves)
        
        for i in range(len(legal_moves)):
            new_state = deepcopy(self.state)
            new_state.push_san(str(legal_moves[i]))
            if self.prob_dist != None:
                self.children.append(MCTS_Node(self, self.player, new_state, self.prob_dist[i]))
            else:
                self.children.append(MCTS_Node(self, self.player, new_state, 1))
                  
    def best_child(self):
        # if player 1 (0) to move, the node with the highest UCB
        # if player 2 (1) to move, the node with the lowest UCB
        best_child = self.children[0]
        
        for child in self.children[1:]:
            if self.player == 0:
                if child.UCB > best_child.UCB:
                    best_child = child
            else:
                if child.UCB < best_child.UCB:
                    best_child = child
        
        return best_child
    
    def simulation(self):
        # choose random moves until terminal state is reached
        # then evaluate board and return value
        if self.is_terminal():
            return self.evaluate_board()
        else:
            move = choice(list(self.state.legal_moves))
            new_state = deepcopy(self.state)
            new_state.push_san(str(move))
            new_node = MCTS_Node(None, self.player, new_state, 1)
            return new_node.simulation()
            
    def evaluate_board(self):
        # +1 - win
        #  0 - draw
        # -1 - loss
        if self.state.is_checkmate() and self.player == 1:
            return 1.0
        elif self.state.is_checkmate() and self.player == 0:
            return -1.0
        else:
            return 0.0
    
    def backpropagate(self, value):
        #   N = N + 1
        #   W = W + v
        #   Q = W / N
        # UCB = Q + 2 x sqrt(ln(parent.N) / self.N)
        if self.parent != None:
            self.parent.backpropagate(value)
            self.N += 1
            self.W += value
            self.Q = self.W / self.N
            self.UCB = self.Q + 2 * sqrt((log(self.parent.N) / self.N)) * self.P
        else:
            self.N += 1
            

class MCTS:
    
    def __init__(self, root, iterations, mode, model):
        self.root = root
        self.iterations = iterations
        self.mode = mode
        self.model = model
    
    def search(self):
        for i in tqdm(range(self.iterations), desc="Making Move...", ncols=100):
            current_node = self.root
            
            if current_node.is_leaf():
                current_node.create_children()
            # Selection - traversing the tree using UCB
            while not current_node.is_leaf():
                current_node = current_node.best_child()
                
            # Expansion - create new child nodes on leaf
            if current_node.is_terminal():
                return current_node.state
            else:
                if current_node.N > 0:
                    if self.model != None:
                        current_node.prob_dist, value = self.use_model(current_node.state)
                    current_node.create_children()
                    current_node = current_node.best_child()
            
            # Simulation - simulate random moves to get a value of the final state
            if self.model == None:
                value = current_node.simulation()
                           
            # Backpropagate - update the values stored in the nodes
            current_node.backpropagate(value)
            
        # Select Move - choose best move depending on mode
            # train - scholastically (exploration), test - deterministically (competition)      
        if self.mode == 'train' and self.model != None:
            if not  self.root.prob_dist:
                self.root.prob_dist = [(child.N / self.N) for child in self.root.children]
            best_move = choices(self.root.children, weights=self.root.prob_dist)
        else: # self.mode == 'test'
            best_move = self.root.children[0]
            
            for child in self.root.children[1:]:
                if child.N > best_move.N:
                    best_move = child
                elif child.N == best_move.N:
                    if child.UCB > best_move.UCB:
                        best_move = child
                                
        return best_move.state
    
    def new_root(self, new_state):
        for child in self.root.children:  
            if child.state == new_state:
                self.root = child
                self.root.parent = None
                break

    def use_model(self, state):
        return self.model.predict(state)






