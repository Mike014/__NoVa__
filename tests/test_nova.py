import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse_command
from formatting import clean_response, split_message
from channel_memory import ChatMemory


class ParserTests(unittest.TestCase):
    def test_search_keeps_case(self):
        parsed = parse_command("!search Best AI Frameworks")
        self.assertEqual(parsed, {"intent": "search_web", "parameters": {"query": "Best AI Frameworks"}})

    def test_visit_keeps_url_case(self):
        parsed = parse_command("Visit webpage https://huggingface.co/Qwen/Qwen2.5")
        self.assertEqual(parsed["parameters"]["url"], "https://huggingface.co/Qwen/Qwen2.5")
        self.assertEqual(parse_command("visita la pagina https://example.com")["intent"], "visit_webpage")

    def test_summarize(self):
        parsed = parse_command("Riassumi questo testo: Ciao Mondo.\nSeconda riga.")
        self.assertEqual(parsed["parameters"]["text"], "Ciao Mondo.\nSeconda riga.")

    def test_change_model_only_at_start(self):
        self.assertEqual(parse_command("switch to Gemma")["parameters"]["model"], "gemma")
        self.assertEqual(parse_command("usa mistral")["intent"], "change_model")
        self.assertEqual(parse_command("io di solito use llama per il backend")["intent"], "unknown")

    def test_other_intents(self):
        self.assertEqual(parse_command("!context")["intent"], "check_context")
        self.assertEqual(parse_command("Che ore sono?")["intent"], "get_time")
        self.assertEqual(parse_command("Scrivimi un hello world in C++")["intent"], "unknown")


class FormattingTests(unittest.TestCase):
    def test_clean_response(self):
        self.assertEqual(clean_response("NoVa: ciao<|eot_id|>"), "ciao")
        self.assertEqual(clean_response("[mike_014__]: ehi"), "ehi")

    def test_clean_response_keeps_every_code_block(self):
        text = "Ecco:\n```py\na = 1\n```\ntesto in mezzo\n```js\nb = 2\n```\nfine"
        self.assertEqual(clean_response(text), "```py\na = 1\n```\n\n```js\nb = 2\n```")

    def test_split_short(self):
        self.assertEqual(split_message("ciao"), ["ciao"])

    def test_split_respects_limit_and_code_fences(self):
        code = "```python\n" + "\n".join(f"print({i})" for i in range(600)) + "\n```"
        chunks = split_message(code, limit=500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)
            self.assertTrue(chunk.startswith("```python"))
            self.assertTrue(chunk.endswith("```"))
        joined = "\n".join(line for chunk in chunks for line in chunk.split("\n") if not line.startswith("```"))
        self.assertEqual(joined.split("\n"), [f"print({i})" for i in range(600)])

    def test_split_very_long_line(self):
        chunks = split_message("x" * 4500)
        self.assertEqual("".join(chunks), "x" * 4500)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))


def record(message_id, channel_id, content, author_id=1, is_bot=0):
    return {
        "message_id": message_id, "guild_id": 10, "channel_id": channel_id, "channel_name": f"c{channel_id}",
        "author_id": author_id, "author_name": f"user{author_id}", "is_bot": is_bot, "content": content,
        "created_at": "2026-10-29T15:00:00+00:00", "jump_url": f"https://discord.com/{message_id}",
    }


class ChatMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = ChatMemory(os.path.join(self.tmp.name, "test.db"))
        self.memory.store([
            record(1, 100, "Ieri ho cucinato la carbonara"),
            record(2, 100, "Il progetto audio procede bene"),
            record(3, 200, "Nel canale privato parliamo della carbonara segreta"),
            record(4, 100, "La carbonara era ottima", author_id=99, is_bot=1),
            record(5, 100, "ti ricordi la carbonara?"),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_skips_duplicates(self):
        self.assertEqual(self.memory.store([record(1, 100, "doppione")]), 0)

    def test_recent_up_to(self):
        ids = [row["message_id"] for row in self.memory.recent(100, 10, up_to=4)]
        self.assertEqual(ids, [1, 2, 4])

    def test_recall_only_in_allowed_channels(self):
        snippets = self.memory.recall("ti ricordi la carbonara?", 100, [100], exclude_ids={5})
        ids = {row["message_id"] for rows in snippets for row in rows}
        self.assertIn(1, ids)
        self.assertNotIn(3, ids)
        self.assertNotIn(4, ids)

    def test_recall_includes_allowed_private_channel(self):
        snippets = self.memory.recall("ti ricordi la carbonara?", 200, [100, 200], exclude_ids={5})
        ids = {row["message_id"] for rows in snippets for row in rows}
        self.assertIn(3, ids)

    def test_edit_and_delete_update_search(self):
        self.memory.update_content(2, "Il progetto backend procede bene")
        self.assertEqual([r["message_id"] for r in self.memory.search("backend", [100])], [2])
        self.memory.delete(2)
        self.assertEqual(self.memory.search("backend", [100]), [])


if __name__ == "__main__":
    unittest.main()
