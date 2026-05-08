"""Base-model continuation seeds.

These prompts are chosen to look like the *middle* of natural text (no chat
template, no instruction framing). A pretrained base model should continue
each one fluently. The set spans encyclopedic prose, journalism, fiction,
code, technical writing, dialogue, correspondence, and reference matter.
"""

from __future__ import annotations

PROMPTS: tuple[str, ...] = (
    # Encyclopedic / Wikipedia-style openers
    "Marie Curie was a Polish-born physicist and chemist whose pioneering research on",
    "The city of Kyoto served as the imperial capital of Japan for more than a thousand years, during which",
    "Photosynthesis is the biochemical process by which green plants and certain other organisms",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and established",
    # News leads
    "WASHINGTON, Reuters - The Federal Reserve announced on Tuesday that it would",
    "TOKYO (AP) - A magnitude 6.4 earthquake struck off the coast of Hokkaido early Wednesday morning, prompting",
    "LONDON - Shares of major European banks fell sharply on Friday after",
    "NEW DELHI - Indian authorities began the largest infrastructure inspection in a decade after",
    # Fiction openings
    "The old lighthouse stood at the edge of the cliff, and every night for the past forty years",
    "She found the letter tucked inside a book she had not opened since college, and the moment her fingers",
    "Detective Halloran knew something was wrong the second he stepped into the apartment, because",
    "On the morning of his eleventh birthday, Tomas woke up to find that his shadow had",
    # Code seeds (raw, no chat formatting)
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
    "class BinarySearchTree:\n    def __init__(self) -> None:\n        self.root = None\n\n    def insert(self, value):\n        ",
    "import numpy as np\n\ndef softmax(logits: np.ndarray) -> np.ndarray:\n    ",
    "// Returns the longest common subsequence of two strings using dynamic programming.\nfunction lcs(a, b) {\n    ",
    # Technical / academic prose
    "In this paper, we introduce a new method for estimating the sample complexity of",
    "The principal contribution of this work is a tight upper bound on the",
    "Recent advances in transformer-based language models have demonstrated that",
    "Throughout the remainder of this section we adopt the notation of Section 2 and assume that",
    # Reference / how-to / instructional
    "To brew a proper pour-over coffee, begin by warming the carafe with hot water, then",
    "When troubleshooting a kernel panic on a Linux server, the first step is to",
    "The fundamental theorem of calculus states that, given a continuous function f on [a, b],",
    "A well-formed regular expression for matching IPv4 addresses must account for",
    # Dialogue / scripts
    "\"You're not seriously suggesting we drive through the storm,\" Maya said, gripping the dashboard. \"It's",
    "The professor leaned back in her chair and folded her hands. \"There are two ways to think about this problem,\" she said. \"The first is",
    # Letters / correspondence
    "Dear Margaret,\n\nIt has been almost six months since I last wrote, and I owe you an explanation. The truth is that",
    "To Whom It May Concern:\n\nI am writing to formally object to the proposed amendment to the lease, on the grounds that",
    # Philosophy / essay openers
    "There is something peculiar about the way we use the word \"freedom\" in everyday speech, because",
    "If we accept the premise that consciousness arises from the integration of information, then",
    # Recipe / domestic
    "For the dough, combine 500 grams of flour, 10 grams of salt, and 7 grams of instant yeast in a large bowl. Then",
    # History / biography
    "By the autumn of 1918, the war had reduced the Habsburg Empire to a patchwork of competing nationalisms, and",
)


assert len(PROMPTS) == 32, f"expected 32 prompts, got {len(PROMPTS)}"
