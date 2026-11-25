"""Lichess puzzle conversion, presentation and answer grading."""
import re

import chess

FILES = 'a b c d e f g h'


def from_lichess(row):
    """Lichess FENs precede the opponent's setup move; solve from the position after it."""
    board = chess.Board(row['FEN'])
    moves = row['Moves'].split()
    setup = chess.Move.from_uci(moves[0])
    last_san = board.san(setup)
    board.push(setup)
    solution = chess.Move.from_uci(moves[1])
    return {'id': row['PuzzleId'], 'game': row['GameId'], 'fen': board.fen(),
            'side': 'White' if board.turn else 'Black', 'last_move_san': last_san,
            'last_move_uci': moves[0], 'solution_uci': moves[1], 'solution_san': board.san(solution),
            'line_uci': moves[1:], 'rating': row['Rating'], 'themes': list(row['Themes'] or [])}


def diagram(fen):
    """ASCII board from the side to move's usual orientation (White at the bottom)."""
    rows = chess.Board(fen).__str__().splitlines()
    return '\n'.join([f'{8 - i}  {r}' for i, r in enumerate(rows)] + ['   ' + FILES])


BOXED = re.compile(r'\\boxed\{([^{}]*)\}')


def extract_boxed(text):
    found = BOXED.findall(text or '')
    return found[-1].strip() if found else None


def parse_move(fen, answer):
    """Accept SAN or UCI, tolerating move numbers, annotations and $-wrapping."""
    if not answer:
        return None
    board = chess.Board(fen)
    text = answer.strip().strip('$').replace('\\text', '').strip('{} ')
    text = re.sub(r'^\d+\s*\.(\.\.)?\s*', '', text)
    text = re.sub(r'[!?]+$', '', text).replace('0-0-0', 'O-O-O').replace('0-0', 'O-O')
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            pass
    return None


def grade(puzzle, answer):
    """Correct if it is the solution move or any mate in one (Lichess accepts all mates)."""
    move = parse_move(puzzle['fen'], answer)
    if move is None:
        return {'answer': answer, 'move_uci': None, 'correct': False}
    board = chess.Board(puzzle['fen'])
    board.push(move)
    correct = move.uci() == puzzle['solution_uci'] or board.is_checkmate()
    return {'answer': answer, 'move_uci': move.uci(), 'correct': correct}
