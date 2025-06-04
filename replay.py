import chess
import chess.pgn
from sb3_contrib import MaskablePPO
from chess_env import ChessEnv

def save_game_to_pgn(moves, filename="replay.pgn"):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Model vs Itself"
    game.headers["White"] = "RL Agent"
    game.headers["Black"] = "RL Agent"
    
    node = game
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            node = node.add_variation(move)
        else:
            print(f"Нелегальный ход: {move_uci}")
            break

    with open(filename, "w") as f:
        f.write(str(game))
    print(f"Сохранено в файл: {filename}")

# ==== Запуск игры ====

env = ChessEnv()
model = MaskablePPO.load("latest_model")

obs, _ = env.reset()
done = False
moves = []

while not done:
    action_mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=action_mask)
    action = action.item()

    move_uci = env.idx_to_move[action]
    moves.append(move_uci)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    env.render()
    print("-" * 30)

print("UCI-moves:", moves)
save_game_to_pgn(moves, "replay.pgn")