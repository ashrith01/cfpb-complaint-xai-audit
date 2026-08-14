"""
Regression tests for narrative cleaning.

clean_narrative runs once over 2M rows and its output is frozen into the
committed dataset, so a silent change here would alter every downstream number
without any step failing. Each case below is one that was actually observed in
the CFPB data and got the regex wrong at some point during development.
"""

from src.data_prep import MIN_CHARS, _dedupe_key, clean_narrative

PAD = " padding to clear the length floor." * 4


def clean(text: str) -> str:
    return clean_narrative(text + PAD)


class TestRedactionRemoval:
    def test_plain_run_removed(self):
        assert "XXXX" not in clean("account XXXX was closed")

    def test_redacted_dates_removed(self):
        for date in ("XX/XX/XXXX", "XX/XX/2024", "XX/XX/24"):
            assert "X" not in clean(f"on {date} they called").replace("padding", "")

    def test_run_glued_to_digits_removed(self):
        """The obvious \\bX{2,}\\b misses these -- no word boundary after a digit."""
        assert "XXXX" not in clean("my account XXXX1234 and card 5678XXXX")
        assert "1234" in clean("my account XXXX1234")

    def test_real_words_containing_xx_survive(self):
        """Dropping the boundary requirement entirely mangles company names."""
        out = clean("I bought gas at EXXON and shopped at TJ MAXX")
        assert "EXXON" in out and "MAXX" in out


class TestCurrencyMasks:
    def test_amount_unwrapped(self):
        assert "$1000.00" in clean("they charged me {$1000.00} today")

    def test_redacted_amount_dropped(self):
        out = clean("they charged me {$XX.XX} in fees")
        assert "{" not in out and "X.X" not in out

    def test_nested_masks(self):
        out = clean("debt of ${ {$3600.00 } } total")
        assert "$3600.00" in out and "{" not in out and "}" not in out

    def test_unterminated_mask_leaves_no_brace(self):
        out = clean("Balance : {$2500.00XXXX Account")
        assert "{" not in out and "XXXX" not in out

    def test_no_doubled_dollar_signs(self):
        assert "$ $" not in clean("exceeds ${$600.00 }, which")


class TestStructure:
    def test_newlines_collapsed(self):
        assert "\n" not in clean("line one\n\nline two\nline three")

    def test_short_narratives_dropped(self):
        assert clean_narrative("too short") == ""

    def test_returned_text_meets_floor(self):
        out = clean("a genuine complaint about my account")
        assert len(out) >= MIN_CHARS

    def test_non_string_input(self):
        assert clean_narrative(None) == ""
        assert clean_narrative(float("nan")) == ""


class TestDedupeKey:
    def test_template_letters_collide_after_redaction_removal(self):
        """The whole point of deduping: mass-filed letters differ only in the
        fields CFPB redacts, so they must collapse to one key once cleaned."""
        a = clean("I dispute the account XXXX opened on XX/XX/2024 with balance {$500.00}")
        b = clean("I dispute the account XXXX opened on XX/XX/2023 with balance {$500.00}")
        assert _dedupe_key(a) == _dedupe_key(b)

    def test_punctuation_and_case_insensitive(self):
        assert _dedupe_key("Hello, World!") == _dedupe_key("hello world")

    def test_genuinely_different_text_differs(self):
        assert _dedupe_key("mortgage complaint") != _dedupe_key("student loan complaint")
