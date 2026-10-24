"""Check the ALFWorld collector's output contract with the SFT stage."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from examples.prompt_agent.openai_alfworld_collector import IncrementalSaver


class SFTDataFormatTests(unittest.TestCase):
    def test_alfworld_success_steps_use_sharegpt_message_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            saver = IncrementalSaver(str(path))
            steps = [
                [
                    {"role": "user", "content": "Task and observation"},
                    {"role": "assistant", "content": "<action>look</action>"},
                ],
            ]
            asyncio.run(saver.save_trajectory_steps(steps))
            self.assertEqual(json.loads(path.read_text()), [{"messages": steps[0]}])
            self.assertEqual(saver.existing_count, 1)
