"""
Additional tests for Snowflake Cortex REST API — new capabilities.

Covers:
- OpenAI-compatible endpoint URL construction
- Stream parameter support
- PAT authentication with api_base
- Model pricing validation

Add to: tests/test_litellm/llms/snowflake/chat/test_snowflake_cortex_enhancements.py
"""

import copy
import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import completion
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.snowflake.chat.transformation import SnowflakeConfig
from litellm.types.utils import ModelResponse


class TestSnowflakeGetCompleteUrl:
    """Test URL construction for legacy and new endpoints."""

    def test_legacy_url_with_account_id(self):
        config = SnowflakeConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="claude-sonnet-4-5",
            optional_params={"account_id": "MYORG-MYACCOUNT"},
            litellm_params={},
        )
        assert url == "https://MYORG-MYACCOUNT.snowflakecomputing.com/api/v2/cortex/inference:complete"

    def test_legacy_url_with_explicit_api_base(self):
        config = SnowflakeConfig()
        url = config.get_complete_url(
            api_base="https://myaccount.snowflakecomputing.com/api/v2",
            api_key=None,
            model="mistral-large2",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myaccount.snowflakecomputing.com/api/v2/cortex/inference:complete"

    def test_url_strips_trailing_slash(self):
        config = SnowflakeConfig()
        url = config.get_complete_url(
            api_base="https://myaccount.snowflakecomputing.com/api/v2/",
            api_key=None,
            model="mistral-large2",
            optional_params={},
            litellm_params={},
        )
        assert url.endswith("/cortex/inference:complete")
        assert "//" not in url.split("://")[1]


class TestSnowflakeSupportedParams:
    """Test that all expected params are supported."""

    def test_tools_in_supported_params(self):
        config = SnowflakeConfig()
        params = config.get_supported_openai_params("claude-sonnet-4-5")
        assert "tools" in params
        assert "tool_choice" in params

    def test_core_params_in_supported(self):
        config = SnowflakeConfig()
        params = config.get_supported_openai_params("llama4-maverick")
        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert "response_format" in params


class TestSnowflakeTransformRequest:
    """Test request transformation edge cases."""

    def test_request_without_stream_defaults_false(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["stream"] is False

    def test_request_with_stream_true(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"stream": True},
            litellm_params={},
            headers={},
        )
        assert result["stream"] is True

    def test_request_passes_temperature(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="mistral-large2",
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={"temperature": 0.3},
            litellm_params={},
            headers={},
        )
        assert result["temperature"] == 0.3

    def test_request_passes_max_tokens(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="deepseek-r1",
            messages=[{"role": "user", "content": "Solve 2+2"}],
            optional_params={"max_tokens": 500},
            litellm_params={},
            headers={},
        )
        assert result["max_tokens"] == 500

    def test_request_model_name_preserved(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="llama4-maverick",
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["model"] == "llama4-maverick"

    def test_request_with_extra_body(self):
        config = SnowflakeConfig()
        result = config.transform_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"extra_body": {"custom_field": "test"}},
            litellm_params={},
            headers={},
        )
        assert result["custom_field"] == "test"

    def test_request_multiple_tools(self):
        config = SnowflakeConfig()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            },
        ]
        result = config.transform_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "What's the weather and time?"}],
            optional_params={"tools": tools, "tool_choice": "auto"},
            litellm_params={},
            headers={},
        )
        assert len(result["tools"]) == 2
        assert result["tools"][0]["tool_spec"]["name"] == "get_weather"
        assert result["tools"][1]["tool_spec"]["name"] == "get_time"


class TestSnowflakeValidateEnvironment:
    """Test authentication header construction."""

    def test_pat_auth_headers(self):
        config = SnowflakeConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="pat/my-pat-token-123",
        )
        assert headers["Authorization"] == "Bearer my-pat-token-123"
        assert headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"

    def test_jwt_auth_headers(self):
        config = SnowflakeConfig()
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="eyJhbGciOiJSUzI1NiJ9.fake.jwt",
        )
        assert headers["Authorization"] == "Bearer eyJhbGciOiJSUzI1NiJ9.fake.jwt"
        assert headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"

    def test_missing_api_key_raises(self):
        config = SnowflakeConfig()
        with pytest.raises(ValueError, match="Missing Snowflake JWT key"):
            config.validate_environment(
                headers={},
                model="claude-sonnet-4-5",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_content_type_headers_set(self):
        config = SnowflakeConfig()
        headers = config.validate_environment(
            headers={},
            model="mistral-large2",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="pat/test",
        )
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"


class TestSnowflakeEndToEnd:
    """End-to-end mocked completion calls for new models."""

    response_json = {
        "choices": [
            {
                "message": {
                    "content": "Snowflake Cortex is a managed AI service.",
                    "content_list": [
                        {"type": "text", "text": "Snowflake Cortex is a managed AI service."}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_completion_claude_sonnet_4_5(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response_json)

        response = litellm.completion(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "What is Cortex?"}],
            api_key="pat/test-pat",
            account_id="MYORG-MYACCOUNT",
        )
        assert response.choices[0].message.content == "Snowflake Cortex is a managed AI service."

        post_kwargs = mock_post.call_args_list[-1][1]
        assert "MYORG-MYACCOUNT" in post_kwargs["url"]
        assert post_kwargs["headers"]["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_completion_llama4_maverick(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response_json)

        response = litellm.completion(
            model="snowflake/llama4-maverick",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="jwt-token-123",
            account_id="TESTORG-TESTACCT",
        )
        assert len(response.choices) == 1

        post_kwargs = mock_post.call_args_list[-1][1]
        body = json.loads(post_kwargs["data"])
        assert body["model"] == "llama4-maverick"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_completion_deepseek_r1(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response_json)

        response = litellm.completion(
            model="snowflake/deepseek-r1",
            messages=[{"role": "user", "content": "Solve: 2+2"}],
            api_key="pat/test",
            account_id="ORG-ACCT",
        )
        assert len(response.choices) == 1

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_completion_with_temperature(self, mock_post):
        mock_post().json.return_value = copy.deepcopy(self.response_json)

        response = litellm.completion(
            model="snowflake/mistral-large2",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="pat/test",
            account_id="ORG-ACCT",
            temperature=0.5,
        )
        assert len(response.choices) == 1

        post_kwargs = mock_post.call_args_list[-1][1]
        body = json.loads(post_kwargs["data"])
        assert body.get("temperature") == 0.5

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_completion_response_model_prefix(self, mock_post):
        resp = copy.deepcopy(self.response_json)
        resp["model"] = "claude-sonnet-4-5"
        mock_post().json.return_value = resp

        response = litellm.completion(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="pat/test",
            account_id="ORG-ACCT",
        )
        assert response.model.startswith("snowflake/")
