import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import main as rag
import torch
from conversation_store import (
    delete_conversations,
    load_conversations,
    save_conversation,
)


class CitationTests(unittest.TestCase):
    def test_selects_only_the_highest_ranked_supporting_passage(self):
        passages = [
            {
                "text": "CNN uses convolutional layers for image features.",
                "source": "notes.pdf",
                "page": 1,
                "rank": 1,
                "score": 0.9,
            },
            {
                "text": "RNN handles sequential data with recurrent connections.",
                "source": "notes.pdf",
                "page": 2,
                "rank": 2,
                "score": 0.8,
            },
            {
                "text": "An unrelated passage about optimization methods.",
                "source": "notes.pdf",
                "page": 3,
                "rank": 3,
                "score": 0.7,
            },
        ]

        self.assertEqual(
            rag.select_citations(
                "Compare CNN and RNN",
                "CNN uses convolutional layers, while RNN handles sequential data.",
                passages,
            ),
            [2],
        )

    def test_answer_generation_returns_without_second_model_pass(self):
        class TestTokenizer:
            eos_token_id = 0

            def apply_chat_template(
                self,
                messages,
                tokenize,
                add_generation_prompt,
            ):
                return (
                    messages[0]["content"]
                    + str(tokenize)
                    + str(add_generation_prompt)
                )

            def __call__(self, prompt, return_tensors, truncation, max_length):
                return {
                    "input_ids": torch.ones(
                        (1, min(len(prompt), max_length)),
                        dtype=torch.long,
                    )
                    if return_tensors == "pt" and truncation
                    else torch.ones((1, len(prompt)), dtype=torch.long)
                }

            def decode(self, tokens, skip_special_tokens=False):
                return (
                    "CNN uses convolutional layers, while RNN handles sequential data."
                    if tokens.numel() and skip_special_tokens
                    else ""
                )

        class TestModel:
            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def eval(self):
                return self

            def generate(self, **kwargs):
                return torch.cat(
                    (kwargs["input_ids"], torch.tensor([[4]])),
                    dim=1,
                )

        passages = [{
            "text": "CNN uses convolutional layers. RNN handles sequential data.",
            "source": "notes.pdf",
            "page": 8,
            "score": 0.9,
        }]

        with (
            patch.object(rag.torch, "no_grad", return_value=nullcontext()),
            patch.object(
                rag,
                "validate_answer",
                side_effect=AssertionError("unexpected second model pass"),
            ),
        ):
            answer = rag.generate_answer(
                "Explain difference between CNN and RNN?",
                passages,
                TestTokenizer(),
                TestModel(),
            )

        self.assertIn("sequential data", answer)
        self.assertIn("[Evidence 1] notes.pdf, page 8", answer)


class ConversationStoreTests(unittest.TestCase):
    def test_conversations_survive_store_reload(self):
        conversation = {
            "title": "How does CNN work?",
            "messages": [
                {"role": "user", "content": "How does CNN work?"},
                {
                    "role": "assistant",
                    "content": "It uses convolutional layers.",
                    "evidence": [
                        {
                            "source": "notes.pdf",
                            "page": 4,
                            "text": "CNN applies convolutional layers.",
                        }
                    ],
                },
            ],
            "updated": 123.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            save_conversation(database, "conversation-1", conversation)

            restored = load_conversations(database)

        self.assertEqual(restored["conversation-1"], conversation)

    def test_delete_conversations_clears_saved_history(self):
        conversation = {"title": "Old chat", "messages": [], "updated": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            save_conversation(database, "conversation-1", conversation)

            delete_conversations(database)

            self.assertEqual(load_conversations(database), {})


if __name__ == "__main__":
    unittest.main()
