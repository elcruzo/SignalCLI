"""Unit tests for JSON validator."""

import pytest
import json

from src.core.exceptions import SchemaValidationError


def test_valid_json_string(json_validator, sample_schema):
    """Test validation of valid JSON string."""
    data = json.dumps(
        {
            "answer": "This is a test answer",
            "confidence": 0.95,
            "topics": ["test", "validation"],
        }
    )

    result = json_validator.validate_and_format(data, sample_schema)

    assert result["answer"] == "This is a test answer"
    assert result["confidence"] == 0.95
    assert result["topics"] == ["test", "validation"]


def test_valid_json_dict(json_validator, sample_schema):
    """Test validation of valid JSON dict."""
    data = {
        "answer": "This is a test answer",
        "confidence": 0.95,
        "topics": ["test", "validation"],
    }

    result = json_validator.validate_and_format(data, sample_schema)

    assert result["answer"] == "This is a test answer"
    assert result["confidence"] == 0.95


def test_missing_required_field(json_validator, sample_schema):
    """Test validation with missing required field."""
    data = {"confidence": 0.95, "topics": ["test"]}

    with pytest.raises(SchemaValidationError):
        json_validator.validate_and_format(data, sample_schema)


def test_invalid_type(json_validator, sample_schema):
    """Test validation with invalid type."""
    data = {
        "answer": "Valid answer",
        "confidence": "not a number",  # Should be number
        "topics": ["test"],
    }

    with pytest.raises(SchemaValidationError):
        json_validator.validate_and_format(data, sample_schema)


def test_string_length_constraint(json_validator, sample_schema):
    """Test string length constraint."""
    long_string = "x" * 1000
    data = {"answer": long_string, "confidence": 0.5}

    result = json_validator.validate_and_format(data, sample_schema)

    # Should be truncated to max_string_length (500)
    assert len(result["answer"]) == json_validator.max_string_length


def test_array_length_constraint(json_validator, sample_schema):
    """Test array length constraint."""
    data = {
        "answer": "Test",
        "confidence": 0.5,
        "topics": [f"topic_{i}" for i in range(50)],
    }

    result = json_validator.validate_and_format(data, sample_schema)

    # Should be truncated to max_array_length (20)
    assert len(result["topics"]) == json_validator.max_array_length


def test_repair_json_missing_quotes(json_validator):
    """Test JSON repair for missing quotes."""
    malformed = '{answer: "test", confidence: 0.5}'

    repaired = json_validator.repair_json(malformed)

    assert repaired is not None
    assert repaired["answer"] == "test"
    assert repaired["confidence"] == 0.5


def test_repair_json_trailing_comma(json_validator):
    """Test JSON repair for trailing comma."""
    malformed = '{"answer": "test", "confidence": 0.5,}'

    repaired = json_validator.repair_json(malformed)

    assert repaired is not None
    assert repaired["answer"] == "test"


def test_invalid_schema(json_validator):
    """Test with invalid schema."""
    invalid_schema = {"type": "invalid_type"}
    data = {"test": "data"}

    with pytest.raises(SchemaValidationError):
        json_validator.validate_and_format(data, invalid_schema)
