import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import chess.engine
from typing import Dict, Tuple, Optional, Any


class ChessEnv(gym.Env):
    """
    Шахматная среда для обучения с подкреплением.
    Использует движок Stockfish для оценки позиций.
    
    Атрибуты:
        board: Текущее состояние шахматной доски
        max_steps: Максимальное количество ходов в эпизоде
        steps: Счетчик сделанных ходов
        engine: Интерфейс к движку Stockfish
        engine_depth: Глубина анализа Stockfish
        all_moves: Список всех возможных UCI-ходов
        move_to_idx: Словарь для преобразования хода в индекс
        idx_to_move: Словарь для преобразования индекса в ход
    """
    
    # Константы для представления доски
    PIECE_TYPES = 6  # 6 типов фигур: пешка, конь, слон, ладья, ферзь, король
    COLORS = 2       # 2 цвета: белые и черные
    BOARD_DIM = 8    # Размер доски 8x8
    
    def __init__(self, stockfish_path: str = "/usr/games/stockfish", engine_depth: int = 12):
        """
        Инициализация шахматной среды.
        
        Args:
            stockfish_path: Путь к исполняемому файлу Stockfish
            engine_depth: Глубина анализа Stockfish (количество полуходов)
        """
        super().__init__()
        self.board = chess.Board()
        self.max_steps = 100
        self.steps = 0

        # Пространство наблюдений: 8x8x12 (12 плоскостей - по 6 типов фигур для каждого цвета)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.BOARD_DIM, self.BOARD_DIM, self.PIECE_TYPES * self.COLORS), 
            dtype=np.float32
        )

        # Генерация всех возможных UCI-ходов и создание словарей для преобразования
        self.all_moves = self._generate_all_uci_moves()
        self.move_to_idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx_to_move = {idx: move for idx, move in enumerate(self.all_moves)}
        self.action_space = spaces.Discrete(len(self.all_moves))

        # Инициализация движка Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine_depth = engine_depth

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Сброс среды в начальное состояние.
        
        Returns:
            Кортеж из:
            - Наблюдение (текущее состояние доски)
            - Пустой словарь (для совместимости с Gymnasium)
        """
        self.board.reset()
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Выполнение одного хода в среде.
        
        Args:
            action: Индекс хода в списке всех возможных ходов
            
        Returns:
            Кортеж из:
            - Наблюдение (новое состояние доски)
            - Награда за ход
            - Флаг завершения эпизода
            - Флаг обрыва эпизода (по достижению max_steps)
            - Информационный словарь
        """
        self.steps += 1
        move_uci = self.idx_to_move[action]
        move = chess.Move.from_uci(move_uci)

        # Проверка допустимости хода
        if move not in self.board.legal_moves:
            return self.get_observation(), -100.0, True, False, {"reason": "illegal move"}

        # Оценка позиции до и после хода
        eval_before = self.evaluate_board()
        self.board.push(move)
        eval_after = self.evaluate_board()

        # Награда - разница в оценке позиции
        reward = eval_after - eval_before

        # Проверка условий завершения игры
        terminated = self.board.is_game_over()
        truncated = self.steps >= self.max_steps

        # Дополнительная награда за результат игры
        if terminated:
            reward += self._calculate_game_result_reward()

        return self.get_observation(), reward, terminated, truncated, {}

    def get_observation(self) -> np.ndarray:
        """
        Преобразование текущего состояния доски в числовой массив.
        
        Returns:
            3D массив numpy размером 8x8x12, где:
            - первые 6 плоскостей - фигуры белых
            - последние 6 плоскостей - фигуры черных
            - 1 на позиции фигуры, 0 - отсутствие фигуры
        """
        planes = np.zeros((self.BOARD_DIM, self.BOARD_DIM, self.PIECE_TYPES * self.COLORS), dtype=np.float32)
        
        # Заполнение плоскостей для каждой клетки доски
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane_idx = self._piece_to_plane(piece)
                row, col = self._square_to_coords(square)
                planes[row, col, plane_idx] = 1
                
        return planes

    def _piece_to_plane(self, piece: chess.Piece) -> int:
        """
        Определение индекса плоскости для заданной фигуры.
        
        Args:
            piece: Шахматная фигура
            
        Returns:
            Индекс плоскости в observation:
            - 0-5 для белых фигур
            - 6-11 для черных фигур
        """
        offset = 0 if piece.color == chess.WHITE else self.PIECE_TYPES
        return offset + (piece.piece_type - 1)

    @staticmethod
    def _square_to_coords(square: chess.Square) -> Tuple[int, int]:
        """
        Преобразование шахматных координат в индексы массива.
        
        Args:
            square: Клетка шахматной доски
            
        Returns:
            Кортеж (строка, столбец) для массива numpy
        """
        row = 7 - (square // 8)  # Инвертируем строки (для отображения как в шахматах)
        col = square % 8
        return row, col

    def _generate_all_uci_moves(self) -> list:
        """
        Генерация всех возможных UCI-ходов, включая превращения пешек.
        
        Returns:
            Отсортированный список всех возможных UCI-ходов
        """
        moves = set()
        
        # Перебираем все возможные комбинации "откуда-куда"
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue  # Пропускаем ходы на месте
                    
                from_rank = chess.square_rank(from_sq)
                to_rank = chess.square_rank(to_sq)
                
                # Проверка на превращение пешки
                is_promotion = (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0)

                if is_promotion:
                    # Генерация всех вариантов превращения
                    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        moves.add(move.uci())
                else:
                    # Обычные ходы без превращения
                    move = chess.Move(from_sq, to_sq)
                    moves.add(move.uci())
                    
        return sorted(moves)

    def action_masks(self) -> np.ndarray:
        """
        Создание маски допустимых ходов для текущей позиции.
        
        Returns:
            Массив numpy с флагами допустимых ходов (True - ход возможен)
        """
        mask = np.zeros(len(self.all_moves), dtype=bool)
        
        # Устанавливаем True для всех легальных ходов
        for move in self.board.legal_moves:
            idx = self.move_to_idx.get(move.uci())
            if idx is not None:
                mask[idx] = True
                
        return mask

    def evaluate_board(self) -> float:
        """
        Оценка текущей позиции с помощью Stockfish.
        
        Returns:
            Оценка позиции в пешках (положительная - преимущество белых)
            Если мат - возвращает +100 или -100 в зависимости от стороны
        """
        limit = chess.engine.Limit(depth=self.engine_depth)
        info = self.engine.analyse(self.board, limit)
        score = info["score"].white()  # Оценка с точки зрения белых
        
        if score.is_mate():
            mate_in = score.mate()
            return 100 if mate_in > 0 else -100  # Мат в пользу белых или черных
        return score.score() / 100.0  # Переводим из сантипешек в пешки

    def _calculate_game_result_reward(self) -> float:
        """
        Расчет дополнительной награды за результат игры.
        
        Returns:
            Награда за победу/поражение/ничью
        """
        result = self.board.result()
        
        if result == "1-0":  # Победа белых
            return 1000 if not self.board.turn else -1000
        elif result == "0-1":  # Победа черных
            return 1000 if self.board.turn else -1000
        return -500  # Ничья

    def render(self, mode: str = 'human') -> None:
        """Вывод текущего состояния доски в консоль."""
        print(self.board)

    def close(self) -> None:
        """Завершение работы и освобождение ресурсов (закрытие Stockfish)."""
        self.engine.quit()


class SelfPlayChessEnv(ChessEnv):
    """
    Среда для самообучения, где противником выступает другая модель.
    
    Атрибуты:
        opponent_model: Модель, которая играет за противника
    """
    
    def __init__(self, opponent_model: Optional[Any] = None, **kwargs):
        """
        Инициализация среды для самообучения.
        
        Args:
            opponent_model: Модель-оппонент (если None, ходы противника не делаются)
            **kwargs: Аргументы для родительского класса ChessEnv
        """
        super().__init__(**kwargs)
        self.opponent_model = opponent_model

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Выполнение хода игрока и ответного хода противника.
        
        Args:
            action: Индекс хода в списке всех возможных ходов
            
        Returns:
            Кортеж из:
            - Наблюдение (новое состояние доски)
            - Награда за ход
            - Флаг завершения эпизода
            - Флаг обрыва эпизода
            - Информационный словарь
        """
        self.steps += 1
        move_uci = self.idx_to_move[action]
        move = chess.Move.from_uci(move_uci)

        # Проверка допустимости хода
        if move not in self.board.legal_moves:
            return self.get_observation(), -100.0, True, False, {"reason": "illegal move"}

        # Оценка позиции до хода
        eval_before = self.evaluate_board()
        self.board.push(move)

        # Проверка завершения игры после хода игрока
        if self.board.is_game_over() or self.steps >= self.max_steps:
            reward = self._calculate_game_result_reward()
            return self.get_observation(), reward, True, self.steps >= self.max_steps, {}

        # Ход противника (если модель предоставлена)
        if self.opponent_model:
            obs = self.get_observation()
            mask = self.action_masks()
            
            # Получение хода от модели-оппонента
            action_opp, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=True)
            opp_move = chess.Move.from_uci(self.idx_to_move[action_opp])
            
            if opp_move in self.board.legal_moves:
                self.board.push(opp_move)
            else:
                # Если противник сделал недопустимый ход - победа игрока
                return self.get_observation(), 100.0, True, False, {"reason": "opponent illegal move"}

        # Оценка позиции после хода противника и расчет награды
        eval_after = self.evaluate_board()
        reward = eval_after - eval_before

        # Проверка условий завершения игры
        terminated = self.board.is_game_over()
        truncated = self.steps >= self.max_steps
        
        # Дополнительная награда за результат игры
        if terminated:
            reward += self._calculate_game_result_reward()

        return self.get_observation(), reward, terminated, truncated, {}