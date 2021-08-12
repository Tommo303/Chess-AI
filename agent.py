from mcts import MCTS_Node, MCTS


class Agent:
    
    def __init__(self, board, player, search_iterations, mode, model):
        self.MCTS = MCTS(MCTS_Node(None, 1-player, board, 1), search_iterations, mode, model) 
    
    def make_move(self, board):
        board = self.MCTS.search()
        self.MCTS.new_root(board)
        return board
        
    def update_tree(self, board):
        if self.MCTS.root.is_leaf():
            self.MCTS.root.create_children()
        self.MCTS.new_root(board)