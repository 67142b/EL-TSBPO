"""Read the task shown by TextWorld without resetting a simulator."""

from functools import lru_cache
import json
from pathlib import Path
import re


@lru_cache(maxsize=8192)
def task_instruction(gamefile: str) -> str:
    """Use the generated game's instruction, including any human annotation.

    The ALFRED trajectory metadata describes the goal but need not contain the
    wording actually used by this generated TextWorld game. Its static `task`
    grammar production is the source of the actor's initial task text.
    """
    game = json.loads(Path(gamefile).read_text())
    blocks = re.findall(r'\bgrammar\s*::\s*"""(.*?)"""\s*;', game['grammar'], re.DOTALL)
    if not blocks:
        raise ValueError(f'Unsupported ALFWorld task grammar: {gamefile}')
    grammar = {}
    for block in blocks:
        block = '\n'.join(line for line in block.splitlines() if not line.lstrip().startswith('#'))
        grammar.update(json.loads(block))
    instructions = set()
    for rule in grammar.get('task', []):
        text = str(rule.get('rhs', '')).strip()
        prefix = 'Your task is to: '
        if not text.startswith(prefix):
            raise ValueError(f'ALFWorld task lacks its instruction prefix: {gamefile}')
        text = text[len(prefix):].strip()
        if not text or any(symbol in text for symbol in ('#', '{', '}')) or rule.get('condition'):
            raise ValueError(f'ALFWorld task instruction is empty or dynamic: {gamefile}')
        instructions.add(text)
    if len(instructions) != 1:
        raise ValueError(f'ALFWorld task must have one deterministic instruction: {gamefile}')
    return instructions.pop()


def root_task_texts(gamefiles):
    return {str(path): task_instruction(str(path)) for path in gamefiles}
