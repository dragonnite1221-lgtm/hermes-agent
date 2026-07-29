from agent.auxiliary_client import _missing_provider_credentials_error


def test_vertex_missing_credentials_error_describes_oauth_not_api_key():
    message = str(_missing_provider_credentials_error("vertex"))

    assert "OAuth credentials" in message
    assert "application-default login" in message
    assert "does not use a static API key" in message
    assert "VERTEX_API_KEY" not in message


def test_api_key_provider_keeps_api_key_guidance():
    message = str(_missing_provider_credentials_error("gemini"))

    assert "GEMINI_API_KEY" in message
