"""Unit tests for preprocess.py cleaning functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from preprocess import (
    clean_article,
    cut_contact_block,
    normalize_whitespace,
    prepend_title,
    split_train_eval,
    strip_emails,
    strip_phones,
    strip_urls,
)

# ---------------------------------------------------------------------------
# cut_contact_block
# ---------------------------------------------------------------------------

class TestCutContactBlock:

    def test_cuts_at_in_case_of_queries(self):
        text = "BMW announced a new model.\n\nIn case of queries, please contact:\nJohn Doe"
        result = cut_contact_block(text)
        assert result == "BMW announced a new model.\n\n"
        assert "In case of queries" not in result

    def test_cuts_at_if_you_have_any_questions(self):
        text = "Great results in 2023.\n\nIf you have any questions, please contact:\nBMW PR"
        result = cut_contact_block(text)
        assert result == "Great results in 2023.\n\n"

    def test_cuts_at_for_further_information(self):
        text = "New technology.\n\nFor further information please contact:\nPress Office"
        result = cut_contact_block(text)
        assert result == "New technology.\n\n"

    def test_cuts_at_in_the_event_of_enquiries(self):
        text = "Launch event.\n\nIn the event of enquiries please contact BMW."
        result = cut_contact_block(text)
        assert result == "Launch event.\n\n"

    def test_case_insensitive(self):
        text = "Body text.\n\nIN CASE OF QUERIES, please contact us."
        result = cut_contact_block(text)
        assert result == "Body text.\n\n"

    def test_uses_rfind_for_last_occurrence(self):
        """If a marker phrase appears in the body AND at the end, cut at the last one."""
        text = (
            "If you have any questions about the iX3, visit bmw.com.\n"
            "More details here.\n\n"
            "If you have any questions, please contact:\nPR Team"
        )
        result = cut_contact_block(text)
        # Should keep the first occurrence (in-text mention) but cut at the last one
        assert "questions about the iX3" in result
        assert "PR Team" not in result

    def test_no_marker_leaves_text_unchanged(self):
        text = "BMW i4 is fully electric. Available worldwide."
        assert cut_contact_block(text) == text

    def test_earliest_marker_wins_among_last_occurrences(self):
        """When multiple markers each appear, cut at the earliest position."""
        text = (
            "Article body.\n\n"
            "In case of queries, contact:\nTeam A\n\n"
            "If you have any questions, contact:\nTeam B"
        )
        result = cut_contact_block(text)
        # "In case of queries" appears at an earlier position than "If you have any questions"
        assert result == "Article body.\n\n"

    def test_does_not_cut_at_corporate_communications(self):
        """Corporate Communications appears in job titles — should NOT be a marker."""
        text = (
            "Head of Corporate Communications, BMW Group France, said: "
            "\"We are excited about the new model.\""
        )
        assert cut_contact_block(text) == text


# ---------------------------------------------------------------------------
# strip_urls
# ---------------------------------------------------------------------------

class TestStripUrls:

    def test_removes_https_url(self):
        text = "Visit https://www.bmwgroup.com/innovation for details."
        assert strip_urls(text) == "Visit  for details."

    def test_removes_http_url(self):
        text = "See http://bmw.com for info."
        assert strip_urls(text) == "See  for info."

    def test_removes_www_url(self):
        text = "Visit www.press.bmwgroup.com for press releases."
        assert strip_urls(text) == "Visit  for press releases."

    def test_preserves_non_url_text(self):
        text = "The BMW iX3 is a fully electric SUV."
        assert strip_urls(text) == text

    def test_removes_youtube_url(self):
        text = "Watch at https://youtu.be/abc123 for the premiere."
        assert strip_urls(text) == "Watch at  for the premiere."

    def test_removes_multiple_urls(self):
        text = "A: https://a.com B: www.b.com end"
        assert strip_urls(text) == "A:  B:  end"


# ---------------------------------------------------------------------------
# strip_emails
# ---------------------------------------------------------------------------

class TestStripEmails:

    def test_removes_standard_email(self):
        text = "Contact: julian.kisch@mini.com for details."
        assert strip_emails(text) == "Contact:  for details."

    def test_removes_email_with_plus(self):
        text = "Email press+releases@bmwgroup.com today."
        assert strip_emails(text) == "Email  today."

    def test_preserves_non_email_text(self):
        text = "BMW M3 Competition with xDrive"
        assert strip_emails(text) == text

    def test_removes_generic_press_email(self):
        text = "Email: presse@bmwgroup.com"
        assert strip_emails(text) == "Email: "


# ---------------------------------------------------------------------------
# strip_phones
# ---------------------------------------------------------------------------

class TestStripPhones:

    def test_removes_international_phone(self):
        text = "Telephone: +49 89 382-38072 for press."
        result = strip_phones(text)
        assert "+49" not in result
        assert "89" not in result
        assert "382-38072" not in result

    def test_removes_phone_with_parens(self):
        text = "Call +49 (89) 382-47564 now."
        result = strip_phones(text)
        assert "+49" not in result
        assert "(89) 382-47564" not in result

    def test_preserves_model_numbers(self):
        text = "The BMW M340i delivers 374 hp."
        assert strip_phones(text) == text

    def test_preserves_short_numbers(self):
        text = "Article +5 new models."
        assert strip_phones(text) == text


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:

    def test_collapses_double_spaces(self):
        assert normalize_whitespace("BMW  iX3  is  great") == "BMW iX3 is great"

    def test_replaces_newlines_with_spaces(self):
        text = "Paragraph one.\nParagraph two."
        assert normalize_whitespace(text) == "Paragraph one. Paragraph two."

    def test_replaces_double_newlines_with_space(self):
        text = "Paragraph one.\n\nParagraph two."
        assert normalize_whitespace(text) == "Paragraph one. Paragraph two."

    def test_replaces_triple_newlines_with_space(self):
        text = "A.\n\n\nB."
        assert normalize_whitespace(text) == "A. B."

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_whitespace("  BMW iX3  ") == "BMW iX3"

    def test_complex_whitespace(self):
        text = "First.  \n\n\n  Second.  \n\n  Third."
        result = normalize_whitespace(text)
        assert result == "First. Second. Third."


# ---------------------------------------------------------------------------
# prepend_title
# ---------------------------------------------------------------------------

class TestPrependTitle:

    def test_basic(self):
        assert prepend_title("Title Here", "Body text.") == "Title Here\nBody text."

    def test_empty_body(self):
        assert prepend_title("Title", "") == "Title\n"


# ---------------------------------------------------------------------------
# clean_article (full pipeline)
# ---------------------------------------------------------------------------

class TestCleanArticle:

    def test_full_pipeline(self):
        title = "BMW Sales Report 2023"
        text = (
            "BMW Group delivered 2.4 million vehicles.\n\n"
            "Visit https://bmw.com for details.\n\n"
            "Contact: press@bmwgroup.com\n"
            "Telephone: +49 89 382-38072\n\n"
            "If you have any questions, please contact:\n"
            "BMW Group Corporate Communications"
        )
        result = clean_article(title, text)
        assert result.startswith("BMW Sales Report 2023\n")
        assert "https://bmw.com" not in result
        assert "press@bmwgroup.com" not in result
        assert "+49 89 382-38072" not in result
        assert "If you have any questions" not in result
        assert "2.4 million vehicles" in result

    def test_article_without_contact_block(self):
        title = "BMW Junior Team"
        text = "Harper, Hesse and Verhagen are BMW M works drivers."
        result = clean_article(title, text)
        assert result == "BMW Junior Team\nHarper, Hesse and Verhagen are BMW M works drivers."

    def test_preserves_fuel_consumption(self):
        """Fuel consumption data IS BMW domain vocabulary — keep it."""
        title = "BMW M3"
        text = (
            "BMW M3 Competition with M xDrive "
            "(fuel consumption combined: 10.1 l/100 km)."
        )
        result = clean_article(title, text)
        assert "fuel consumption combined: 10.1 l/100 km" in result


# ---------------------------------------------------------------------------
# split_train_eval
# ---------------------------------------------------------------------------

class TestSplitTrainEval:

    def test_90_10_split(self):
        articles = [{"article_id": f"T{i:07d}EN", "text": f"Article {i}"} for i in range(100)]
        train, eval_ = split_train_eval(articles)
        assert len(train) == 90
        assert len(eval_) == 10

    def test_no_overlap(self):
        articles = [{"article_id": f"T{i:07d}EN", "text": f"Article {i}"} for i in range(100)]
        train, eval_ = split_train_eval(articles)
        train_ids = {a["article_id"] for a in train}
        eval_ids = {a["article_id"] for a in eval_}
        assert train_ids.isdisjoint(eval_ids)

    def test_no_loss(self):
        articles = [{"article_id": f"T{i:07d}EN", "text": f"Article {i}"} for i in range(100)]
        train, eval_ = split_train_eval(articles)
        assert len(train) + len(eval_) == len(articles)

    def test_same_seed_same_split(self):
        articles = [{"article_id": f"T{i:07d}EN", "text": f"Article {i}"} for i in range(50)]
        train1, _ = split_train_eval(articles)
        train2, _ = split_train_eval(articles)
        assert [a["article_id"] for a in train1] == [a["article_id"] for a in train2]

    def test_different_seed_different_split(self):
        articles = [{"article_id": f"T{i:07d}EN", "text": f"Article {i}"} for i in range(50)]
        train1, _ = split_train_eval(articles, seed=42)
        train2, _ = split_train_eval(articles, seed=99)
        ids1 = [a["article_id"] for a in train1]
        ids2 = [a["article_id"] for a in train2]
        assert ids1 != ids2
