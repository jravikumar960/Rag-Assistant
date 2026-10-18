import tempfile
import unittest
import ast
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
    def _run_generation(self, question, passage, generated):
        class TestTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return messages[0]["content"] + str(tokenize) + str(add_generation_prompt)

            def __call__(self, prompt, return_tensors, truncation, max_length):
                return {"input_ids": torch.ones((1, min(len(prompt), max_length)), dtype=torch.long)}

            def decode(self, tokens, skip_special_tokens=False):
                return generated if tokens.numel() and skip_special_tokens else ""

        class TestModel:
            def __init__(self):
                self.generate_calls = 0

            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.generate_calls += 1
                return torch.cat((kwargs["input_ids"], torch.tensor([[4]])), dim=1)

        model = TestModel()
        with (
            patch.object(rag.torch, "no_grad", return_value=nullcontext()),
            patch.object(
                rag,
                "validate_answer",
                side_effect=AssertionError("unexpected second model pass"),
            ),
        ):
            answer = rag.generate_answer(
                question,
                passage if isinstance(passage, list) else [passage],
                TestTokenizer(),
                model,
            )
        return answer, model.generate_calls

    def test_unresolvable_citation_does_not_display_first_passage(self):
        class FakeStreamlit:
            def __init__(self):
                self.markdown_calls = []
                self.warning_calls = []
                self.popover_calls = []

            def markdown(self, value):
                self.markdown_calls.append(value)

            def warning(self, value):
                self.warning_calls.append(value)

            def columns(self, *_args, **_kwargs):
                raise AssertionError("citation UI should not be rendered")

            def popover(self, *_args, **_kwargs):
                self.popover_calls.append(_args)
                raise AssertionError("citation UI should not be rendered")

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app_tree = ast.parse(app_path.read_text(encoding="utf-8"))
        function_node = next(
            node for node in app_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "display_answer_with_sources"
        )
        test_st = FakeStreamlit()
        namespace = {"re": __import__("re"), "st": test_st}
        exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(app_path), "exec"), namespace)

        passages = [{"source": "first.pdf", "page": 1, "text": "First retrieved passage"}]
        namespace["display_answer_with_sources"](
            "Answer text.\n\nSources: missing.pdf, page 99", passages,
        )

        self.assertEqual(test_st.markdown_calls, ["Answer text."])
        self.assertEqual(test_st.warning_calls, ["Source could not be verified for this answer."])
        self.assertFalse(test_st.popover_calls)

    def test_intentional_abstention_without_citation_has_no_source_warning(self):
        class FakeStreamlit:
            def __init__(self):
                self.markdown_calls = []
                self.warning_calls = []

            def markdown(self, value):
                self.markdown_calls.append(value)

            def warning(self, value):
                self.warning_calls.append(value)

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app_tree = ast.parse(app_path.read_text(encoding="utf-8"))
        function_node = next(
            node for node in app_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "display_answer_with_sources"
        )
        test_st = FakeStreamlit()
        namespace = {"re": __import__("re"), "st": test_st}
        exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(app_path), "exec"), namespace)

        namespace["display_answer_with_sources"](
            "I couldn't find sufficient direct evidence in the retrieved PDF passages to answer this question.",
            [],
        )

        self.assertEqual(len(test_st.markdown_calls), 1)
        self.assertEqual(test_st.warning_calls, [])

    def test_absent_answer_is_rejected_without_citation(self):
        answer, calls = self._run_generation(
            "What is CNN?",
            {"text": "RNN is a recurrent neural network.", "source": "notes.pdf", "page": 3},
            "CNN is a convolutional neural network.",
        )
        self.assertIn("couldn't verify", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_lexically_similar_but_irrelevant_passage_is_rejected(self):
        answer, calls = self._run_generation(
            "What are the disadvantages of CNN?",
            {
                "text": "CNN is a powerful method for image classification.",
                "source": "notes.pdf",
                "page": 4,
            },
            "CNN is a powerful method for image classification.",
        )
        self.assertIn("couldn't verify", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_direct_answer_uses_exact_sentence_and_correct_pdf_page(self):
        sentence = "CNN is a neural network that uses convolutional layers."
        answer, calls = self._run_generation(
            "What is CNN?",
            {"text": sentence, "source": "notes.pdf", "page": 8},
            sentence,
        )
        self.assertEqual(answer, sentence + "\n\nSources: [Evidence 1] notes.pdf, page 8")
        self.assertEqual(calls, 1)

    def test_random_forest_definition_accepts_pdf_bullet_and_cites_actual_page(self):
        sentence = "Random Forest is a technique that uses ensemble learning, that combines many weak classifiers to provide solutions to complex problems."
        passage = "\uf097 " + sentence
        answer, calls = self._run_generation(
            "What is Random Forest?",
            {
                "text": "What is the Random Forest Algorithm? \uf097 " + sentence,
                "source": "811790898-Random-Forest-Algorithm.pdf",
                "page": 9,
            },
            sentence,
        )
        self.assertEqual(
            answer,
            sentence + "\n\nSources: [Evidence 1] 811790898-Random-Forest-Algorithm.pdf, page 9",
        )
        self.assertEqual(calls, 1)
        self.assertEqual(rag.select_citations("What is Random Forest?", sentence, [{"text": passage}]), [1])

    def test_random_forest_definition_extracts_exact_evidence_if_model_paraphrases(self):
        sentence = "Random Forest is a technique that uses ensemble learning, that combines many weak classifiers to provide solutions to complex problems."
        answer, calls = self._run_generation(
            "What is Random Forest?",
            {
                "text": "What is the Random Forest Algorithm? \uf097 " + sentence,
                "source": "811790898-Random-Forest-Algorithm.pdf",
                "page": 9,
            },
            "Random Forest combines several weak classifiers.",
        )
        self.assertEqual(
            answer,
            sentence + "\n\nSources: [Evidence 1] 811790898-Random-Forest-Algorithm.pdf, page 9",
        )
        self.assertEqual(calls, 1)

    def test_random_forest_definition_still_abstains_when_unsupported(self):
        answer, calls = self._run_generation(
            "What is Random Forest?",
            {"text": "CNN is a neural network that uses convolutional layers.", "source": "notes.pdf", "page": 4},
            "Random Forest is an ensemble method.",
        )
        self.assertIn("couldn't verify", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_explain_random_forest_uses_definition_only_when_exactly_supported(self):
        sentence = "Random Forest is a technique that combines decision trees."
        answer, calls = self._run_generation(
            "Explain Random Forest",
            {"text": sentence, "source": "forest.pdf", "page": 9},
            sentence,
        )
        self.assertEqual(answer, sentence + "\n\nSources: [Evidence 1] forest.pdf, page 9")
        self.assertEqual(calls, 1)

    def test_disadvantages_and_vanishing_gradient_require_exact_supported_sentences(self):
        limitation = "The main limitation of Random Forest is that it requires many trees."
        limitation_answer, _ = self._run_generation(
            "What are disadvantages of Random Forest?",
            {"text": limitation, "source": "notes.pdf", "page": 9},
            limitation,
        )
        self.assertIn("[Evidence 1] notes.pdf, page 9", limitation_answer)

        gradient = "Vanishing gradients can prevent the network from being trained properly."
        gradient_answer, _ = self._run_generation(
            "How do vanishing gradients affect training?",
            {"text": gradient, "source": "notes.pdf", "page": 10},
            gradient,
        )
        self.assertIn("[Evidence 1] notes.pdf, page 10", gradient_answer)

    def test_comparison_parser_extracts_entities_and_ignores_output_instructions(self):
        self.assertEqual(
            rag._requested_comparison("Compare CNN and RNN. Show the exact sources."),
            ("CNN", "RNN"),
        )
        self.assertEqual(
            rag._requested_comparison("Compare Random Forest with Decision Trees show sources"),
            ("Random Forest", "Decision Trees"),
        )
        self.assertEqual(
            rag._requested_comparison("Compare CNN versus RNN"),
            ("CNN", "RNN"),
        )

    def test_supported_cnn_rnn_comparison_cites_exact_page_34_sentence(self):
        sentence = (
            "For example, convolutional neural networks (CNNs) perform exceptionally well "
            "in image processing, whereas Recurrent Neural Networks (RNNs) perform well "
            "in text and voice processing."
        )
        answer, calls = self._run_generation(
            "Compare CNN and RNN. Show the exact sources.",
            {
                "text": "Advantages of feedforward neural networks. " + sentence,
                "source": "DEEP LEARNING ( R20A06610 ).pdf",
                "page": 34,
            },
            sentence,
        )
        self.assertEqual(
            answer,
            sentence + "\n\nSources: [Evidence 1] DEEP LEARNING ( R20A06610 ).pdf, page 34",
        )
        self.assertEqual(calls, 1)

    def test_supported_random_forest_decision_tree_comparison_cites_page_30(self):
        sentence = (
            "Difference between Decision Tree and Random Forest Hence, there is a conclusion "
            "that random forests are much more successful than decision trees only if the "
            "trees are diverse and acceptable."
        )
        answer, calls = self._run_generation(
            "compare Random Forest with Decision Trees",
            {
                "text": sentence,
                "source": "811790898-Random-Forest-Algorithm.pdf",
                "page": 30,
            },
            sentence,
        )
        self.assertEqual(
            answer,
            sentence + "\n\nSources: [Evidence 1] 811790898-Random-Forest-Algorithm.pdf, page 30",
        )
        self.assertEqual(calls, 1)

    def test_random_forest_comparison_falls_back_to_exact_page_30_evidence(self):
        sentence = (
            "Difference between Decision Tree and Random Forest Hence, there is a conclusion "
            "that random forests are much more successful than decision trees only if the "
            "trees are diverse and acceptable."
        )
        passage = {
            "text": sentence,
            "source": "811790898-Random-Forest-Algorithm.pdf",
            "page": 30,
        }
        for generated in (
            "Random forests are more successful than decision trees when the trees are diverse and acceptable.",
            "INSUFFICIENT EVIDENCE",
        ):
            with self.subTest(generated=generated):
                answer, calls = self._run_generation(
                    "Compare Random Forest with Decision Trees",
                    passage,
                    generated,
                )
                self.assertEqual(
                    answer,
                    sentence + "\n\nSources: [Evidence 1] 811790898-Random-Forest-Algorithm.pdf, page 30",
                )
                self.assertEqual(calls, 1)

    def test_comparison_abstains_when_only_one_entity_is_supported(self):
        sentence = "CNNs perform exceptionally well in image processing."
        answer, calls = self._run_generation(
            "Compare CNN and RNN",
            {"text": sentence, "source": "vision.pdf", "page": 4},
            sentence,
        )
        self.assertIn("couldn't verify a direct comparison", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_comparison_rejects_both_entities_without_explicit_contrast(self):
        sentence = "CNN and RNN are neural network types."
        answer, calls = self._run_generation(
            "Compare CNN and RNN",
            {"text": sentence, "source": "deep-learning.pdf", "page": 8},
            sentence,
        )
        self.assertIn("couldn't verify a direct comparison", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_comparison_rejects_unrelated_evidence_with_overlapping_keywords(self):
        sentence = "The syllabus glossary lists CNN, RNN, Random Forest, and Decision Trees as keywords."
        answer, calls = self._run_generation(
            "Compare CNN and RNN",
            {"text": sentence, "source": "syllabus.pdf", "page": 2},
            sentence,
        )
        self.assertIn("couldn't verify a direct comparison", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_supported_comparison_uses_only_one_generation_pass(self):
        sentence = "CNNs process images, whereas RNNs process sequences."
        answer, calls = self._run_generation(
            "Compare CNN and RNN",
            {"text": sentence, "source": "models.pdf", "page": 12},
            sentence,
        )
        self.assertIn(sentence, answer)
        self.assertEqual(calls, 1)

    def test_random_forest_disadvantages_with_output_instruction_cites_page_37(self):
        sentence = (
            "The main limitation of random forest is that due to a large number of trees "
            "the algorithm takes a long time to train which makes it slow and ineffective "
            "for real-time predictions."
        )
        passage = (
            "\uf0b7 " + sentence + " \uf0b7 In general, these algorithms are fast to train but quite slow "
            "to create predictions once they are trained. In most real-world applications, "
            "the random forest algorithm is fast enough but there may be situations where "
            "run-time performance is critical, and alternative approaches would be more suitable."
        )
        answer, calls = self._run_generation(
            "What are the disadvantages of Random Forest? Show the exact supporting passage and PDF page number.",
            [
                {
                    "text": "Advantages and Disadvantages of Random Forest. Random forest helps overcome overfitting.",
                    "source": "811790898-Random-Forest-Algorithm.pdf",
                    "page": 36,
                },
                {
                    "text": passage,
                    "source": "811790898-Random-Forest-Algorithm.pdf",
                    "page": 37,
                },
            ],
            "Random Forest is slow to train and can be ineffective for real-time use.",
        )
        self.assertEqual(
            answer,
            sentence + "\n\nSources: [Evidence 2] 811790898-Random-Forest-Algorithm.pdf, page 37",
        )
        self.assertEqual(calls, 1)

    def test_unsupported_paris_population_abstains_without_citation(self):
        answer, calls = self._run_generation(
            "What is the population of Paris?",
            {
                "text": "Random Forest is a technique that combines decision trees.",
                "source": "811790898-Random-Forest-Algorithm.pdf",
                "page": 9,
            },
            "INSUFFICIENT EVIDENCE",
        )
        self.assertIn("couldn't find sufficient direct evidence", answer)
        self.assertNotIn("Sources:", answer)
        self.assertEqual(calls, 1)

    def test_select_citations_requires_exact_sentence_and_requested_relation(self):
        self.assertEqual(
            rag.select_citations(
                "What are the disadvantages of CNN?",
                "CNN is a powerful method for image classification.",
                [{
                    "text": "CNN is a powerful method for image classification.",
                    "source": "notes.pdf",
                    "page": 1,
                }],
            ),
            [],
        )
        self.assertEqual(
            rag.select_citations(
                "What is CNN?",
                "CNN is a neural network that uses convolutional layers.",
                [{
                    "text": "CNN is a neural network that uses convolutional layers.",
                    "source": "notes.pdf",
                    "page": 8,
                }],
            ),
            [1],
        )

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
