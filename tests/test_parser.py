"""Tests for the resilient structured-output parser."""

import pytest

from domain.models import ReviewDecision
from infrastructure.parser import ParseError, StructuredOutputParser


@pytest.fixture
def parser():
    return StructuredOutputParser(max_retries=2)


def test_strips_think_tags_and_fences(parser):
    raw = """<think>
I need to produce JSON only.
</think>
Here is the result:
```json
{
  "verdict": "pass",
  "score": 90,
  "feedback": "well grounded",
  "missing_elements": [],
  "required_changes": []
}
```"""
    decision = parser.parse(raw, ReviewDecision)
    assert decision.verdict == "pass"
    assert decision.score == 90
    assert decision.feedback == "well grounded"


def test_plain_json_without_fences(parser):
    raw = '{"verdict": "revise", "score": 50, "feedback": "add sources", "missing_elements": [], "required_changes": ["add citations"]}'
    decision = parser.parse(raw, ReviewDecision)
    assert decision.verdict == "revise"
    assert decision.score == 50


def test_json_embedded_in_prose(parser):
    raw = (
        "The model decided as follows: {\"verdict\": \"pass\", \"score\": 80, "
        "\"feedback\": \"ok\", \"missing_elements\": [], \"required_changes\": []} end."
    )
    decision = parser.parse(raw, ReviewDecision)
    assert decision.verdict == "pass"


def test_invalid_json_raises_parse_error(parser):
    with pytest.raises(ParseError):
        parser.parse("this is not json at all", ReviewDecision)


def test_missing_required_field_raises(parser):
    raw = '{"verdict": "pass", "score": 90}'  # missing feedback
    with pytest.raises(ParseError):
        parser.parse(raw, ReviewDecision)


def test_parse_or_none_returns_none_on_failure(parser):
    assert parser.parse_or_none("garbage", ReviewDecision) is None


def test_score_out_of_range_raises(parser):
    raw = '{"verdict": "pass", "score": 101, "feedback": "x", "missing_elements": [], "required_changes": []}'
    with pytest.raises(ParseError):
        parser.parse(raw, ReviewDecision)
