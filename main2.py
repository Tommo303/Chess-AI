import chess
from agent import Agent
import pandas as pd
from _thread import start_new_thread


game_history = []


def self_play(mode, tree_depth, model1, model2):
    global game_history 
    
    # Initialise local variables, including the board to be played on
    training_set = []
    board = chess.Board()
    move_count = 0
    
    # Initialise 2 agents to play against each other
    player_0 = Agent(board=board, player=0, 
                     search_iterations=tree_depth, 
                     mode=mode, 
                     model=model1)
    player_1 = Agent(board=board, 
                     player=1, 
                     search_iterations=tree_depth, 
                     mode=mode, 
                     model=model2)
    
    # Play game - agents take it in turns making moves
    while not board.is_game_over():
        board = player_0.make_move(board)
        player_1.update_tree(board)
        move_count += 1
        training_set.append([board.fen(), [child.N / (player_0.MCTS.root.N - 1) for child in player_0.MCTS.root.children]])
        board = player_1.make_move(board)
        player_0.update_tree(board)
        move_count += 1
        training_set.append([board.fen(), [child.N / (player_1.MCTS.root.N - 1) for child in player_0.MCTS.root.children]])
    
    # Determine the result of the game
    if board.is_checkmate() and move_count % 2 == 1:
        result = 1
    elif board.is_checkmate() and move_count % 2 == 0:
        result = -1
    else:
        result = 0
    
    # Update training_set with result of game
    for element in training_set:
        element.append(result)
        
    # Add training set to end of game_history
    game_history = [*game_history, *training_set]
        
    print()
    print(board)
    print(result)


for i in range(10):
    print(f'Starting Game {i}...')
    start_new_thread(self_play, ('train', 500, None, None))
    

df = pd.DataFrame(game_history)
df.to_csv('./game_history.csv')


    
