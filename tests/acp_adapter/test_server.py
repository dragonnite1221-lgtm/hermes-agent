"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    ImageContentBlock,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.model_catalog import ACP_MAX_MODELS_PER_PROVIDER
from acp_adapter.server import (
    HermesACPAgent,
    HERMES_VERSION,
)
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [(mode.id, mode.name) for mode in resp.modes.available_modes] == [
        ("default", "Default"),
        ("accept_edits", "Accept Edits"),
        ("dont_ask", "Don't Ask"),
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION




    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build_payload,
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        assert [model.name for model in resp.models.available_models] == [
            "Anthropic · claude-sonnet-4-6",
            "OpenAI Codex · gpt-5.4",
            "OpenAI Codex · gpt-5.4-mini",
        ]
        assert resp.models.available_models[1].description is not None
        assert "current" in resp.models.available_models[1].description
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )
        build_payload.assert_called_once_with(
            picker_context,
            explicit_only=True,
            include_unconfigured=False,
            picker_hints=False,
            canonical_order=True,
            pricing=False,
            capabilities=False,
            refresh=False,
            probe_custom_providers=False,
            probe_current_custom_provider=False,
            max_models=ACP_MAX_MODELS_PER_PROVIDER,
        )



    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None


    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name", ["resume_session", "load_session"])
    async def test_session_op_fails_explicitly_when_history_unavailable(
        self, tmp_path, monkeypatch, method_name
    ):
        """A DB error while loading a session's history must surface as an
        explicit failure on BOTH resume_session and load_session, not a
        silent success -- resume_session's pre-fix fallback was worse
        ("not found, creating new" empty session reported as a successful
        resume), but load_session likewise returned a successful,
        empty-history load rather than the actual failure.
        """
        db = SessionDB(tmp_path / "state.db")
        manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"), db=db)
        acp_agent = HermesACPAgent(session_manager=manager)

        new_resp = await acp_agent.new_session(cwd=str(tmp_path))
        state = manager.get_session(new_resp.session_id)
        state.history.append({"role": "user", "content": "do not lose me"})
        manager.save_session(state.session_id)

        with manager._lock:
            del manager._sessions[state.session_id]

        def _boom(*args, **kwargs):
            raise RuntimeError("db timeout")

        monkeypatch.setattr(db, "get_messages_as_conversation", _boom)

        with pytest.raises(acp.RequestError):
            await getattr(acp_agent, method_name)(
                cwd=str(tmp_path), session_id=new_resp.session_id
            )

    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result["configOptions"] == []





# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_audio_only_prompt_actually_reaches_the_agent(self, agent, mock_manager):
        """An audio-only prompt must still invoke run_conversation.

        _extract_text(prompt) is "" for an audio-only prompt (it only sees
        real TextContentBlocks), and _content_blocks_to_openai_user_content
        collapses an audio placeholder-only result to a plain string, not a
        list. The prompt() empty-content guard checked
        "isinstance(user_content, list) and user_content", which is False
        for that string -- so an audio-only prompt was rejected as empty
        and never reached run_conversation at all, even after the
        placeholder fix made the CONVERTER stop returning a truly empty
        string.
        """
        from acp.schema import AudioContentBlock

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        run_calls = []

        def _run(*args, **kwargs):
            run_calls.append(kwargs)
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt_resp = await agent.prompt(
            prompt=[AudioContentBlock(type="audio", data="aGVsbG8=", mimeType="audio/wav")],
            session_id=resp.session_id,
        )

        assert isinstance(prompt_resp, PromptResponse)
        assert len(run_calls) == 1
        assert "audio/wav" in str(run_calls[0].get("user_message"))

    @pytest.mark.asyncio
    async def test_whitespace_only_text_prompt_is_rejected_not_dispatched(self, agent, mock_manager):
        """A prompt made of nothing but a whitespace-only TextContentBlock must not reach
        run_conversation. _content_blocks_to_openai_user_content collapses an all-text prompt
        to a joined string built straight from block.text, so a "   " block yields a
        truthy-but-blank user_content -- the guard that lets a real non-text placeholder
        (e.g. the audio-only one) through despite an empty user_text must not also let this
        blank text-only turn through.
        """
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        run_calls = []

        def _run(*args, **kwargs):
            run_calls.append(kwargs)
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt_resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="   ")],
            session_id=resp.session_id,
        )

        assert isinstance(prompt_resp, PromptResponse)
        assert prompt_resp.stop_reason == "end_turn"
        assert run_calls == []

    @pytest.mark.asyncio
    async def test_queued_image_prompt_preserves_attachment_data(self, agent):
        """A prompt that arrives while a turn is already running gets
        queued for the next turn. If it carries an image, the queued item
        must retain the actual image bytes -- not just a "[Image
        attachment]" text placeholder -- or the attachment is silently
        dropped once the queued turn finally runs (see the drain loop that
        replays ``state.queued_prompts`` as a fresh ``self.prompt(...)``
        call).
        """
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.is_running = True  # simulate an in-flight turn

        image_block = ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png")
        prompt = [
            TextContentBlock(type="text", text="look at this"),
            image_block,
        ]

        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert len(state.queued_prompts) == 1
        queued = state.queued_prompts[0]
        assert queued != "[Image attachment]"
        # The queued item must still carry the actual image data somewhere,
        # not just a text summary that stands in for it.
        blocks = queued if isinstance(queued, list) else [queued]
        assert any(
            getattr(block, "data", None) == "aGVsbG8="
            or (isinstance(block, dict) and block.get("data") == "aGVsbG8=")
            for block in blocks
        )

    @pytest.mark.asyncio
    async def test_queued_resource_link_is_snapshotted_not_reread_later(self, agent, tmp_path):
        """A resource_link block only carries a URI. If it's queued as-is,
        the drain loop's replay re-reads that URI whenever the queued turn
        actually runs -- possibly long after the file was modified. The
        queued item must instead be a self-contained snapshot of the
        file's content AT THE TIME IT WAS QUEUED.
        """
        attached = tmp_path / "notes.md"
        attached.write_text("original content", encoding="utf-8")

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.is_running = True  # simulate an in-flight turn

        prompt = [
            TextContentBlock(type="text", text="read this"),
            ResourceContentBlock(
                type="resource_link",
                name="notes.md",
                uri=attached.as_uri(),
                mimeType="text/markdown",
            ),
        ]

        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)
        assert isinstance(resp, PromptResponse)
        assert len(state.queued_prompts) == 1

        # The file changes AFTER queuing but BEFORE the queued turn runs.
        attached.write_text("MUTATED AFTER QUEUING", encoding="utf-8")

        queued = state.queued_prompts[0]
        blocks = queued if isinstance(queued, list) else [queued]
        resource_blocks = [b for b in blocks if hasattr(b, "resource")]
        assert resource_blocks, "queued resource_link was not snapshotted into an embedded resource"
        snapshotted_text = resource_blocks[0].resource.text
        assert snapshotted_text == "original content"
        assert "MUTATED" not in snapshotted_text

    @pytest.mark.asyncio
    async def test_queued_post_interrupt_correction_keeps_the_rewritten_text(
        self, agent, monkeypatch
    ):
        """A post-cancel correction ("stop and send") gets its text rewritten
        to include the cancelled request BEFORE the turn-claim lock is
        taken. If another prompt starts running in that gap, this one is
        queued instead of run immediately -- and the queued item must still
        carry the REWRITTEN text (cancelled request + correction), not just
        the bare new text, or the attached context from the salvage path is
        silently dropped once the queued turn replays.
        """
        import acp_adapter.server as server_module

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)

        original_take = server_module._take_interrupted_prompt

        def _take_and_race(state_arg):
            # Simulate another request claiming the turn in the gap
            # between this consuming the interrupted prompt (while still
            # idle) and _claim_turn_or_queue's later lock acquisition.
            idle, interrupted = original_take(state_arg)
            state_arg.is_running = True
            return idle, interrupted

        monkeypatch.setattr(server_module, "_take_interrupted_prompt", _take_and_race)

        state.interrupted_prompt_text = "please refactor the auth module"

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="actually use the new schema")],
            session_id=new_resp.session_id,
        )

        assert isinstance(resp, PromptResponse)
        assert len(state.queued_prompts) == 1
        queued = state.queued_prompts[0]
        assert isinstance(queued, str)
        assert "please refactor the auth module" in queued
        assert "actually use the new schema" in queued

    @pytest.mark.asyncio
    async def test_queued_image_prompt_replay_actually_delivers_attachment(self, agent):
        """End-to-end: the queued image prompt must reach ``run_conversation``
        with its attachment intact when the drain loop replays it, not just
        sit unmodified in ``state.queued_prompts``.

        A test that only inspects the queue would stay green even if the
        drain loop's replay path silently converted the content back to
        text before calling ``run_conversation`` again. Drive a real first
        turn, queue the image prompt while it is genuinely in flight, let
        the first turn finish, and assert the SECOND ``run_conversation``
        call actually received the image data.
        """
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)

        calls: list[dict] = []
        first_call_started = threading.Event()
        release_first_call = threading.Event()

        def _run(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                first_call_started.set()
                assert release_first_call.wait(timeout=5), "test deadlocked"
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        first_task = asyncio.create_task(
            agent.prompt(
                prompt=[TextContentBlock(type="text", text="first turn")],
                session_id=new_resp.session_id,
            )
        )
        await asyncio.get_event_loop().run_in_executor(None, first_call_started.wait, 5)
        assert state.is_running is True  # the first turn is genuinely in flight

        image_block = ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png")
        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="look at this"), image_block],
            session_id=new_resp.session_id,
        )
        assert isinstance(resp, PromptResponse)
        assert len(state.queued_prompts) == 1

        release_first_call.set()
        await first_task
        # The drain loop's recursive self.prompt(...) call runs on the same
        # event loop but still dispatches through run_in_executor; give it a
        # turn to complete.
        for _ in range(50):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.05)

        assert len(calls) == 2
        # Whatever shape run_conversation's user_message takes (OpenAI-style
        # multimodal content), the base64 image payload must survive into it.
        assert "aGVsbG8=" in json.dumps(calls[1].get("user_message"), default=str)

    @pytest.mark.asyncio
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id

    @pytest.mark.asyncio
    async def test_empty_messages_list_replaces_stale_history(self, agent, mock_manager):
        """``run_conversation`` returning ``messages=[]`` clears the ACP transcript instead of
        leaving the previous turn's history in place (#10844)."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.history = [{"role": "user", "content": "old"}]
        state.agent.run_conversation = MagicMock(return_value={"final_response": "done", "messages": []})
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id)

        assert state.history == []

    @pytest.mark.asyncio
    async def test_is_running_resets_and_queued_prompt_survives_real_transport_failure(
        self, agent, mock_manager, monkeypatch
    ):
        """Core delivery-failure invariant, exercised over a REAL ACP
        JSON-RPC connection instead of a mocked ``session_update`` -- a
        mock never touches ``acp.agent.connection.AgentSideConnection`` /
        ``MessageSender``, the actual code whose broken-pipe behavior this
        fix depends on (per AGENTS.md: I/O-touching fixes need one real
        path, mocks alone hide integration bugs).

        Wires a genuine ``AgentSideConnection`` to a pipe whose read end is
        closed immediately, so every outbound write fails for real, queues
        a prompt before calling ``agent.prompt()``, and asserts the full
        guarantee against the real transport in one shot: the call raises,
        ``is_running`` resets to False, and the queued prompt survives
        instead of being silently dropped.

        This also exercises ``_session_update_or_raise``'s timeout bound:
        once ``MessageSender._loop()`` dies from the FIRST failed write, ANY
        FURTHER ``session_update`` on that same connection silently hangs
        instead of raising (confirmed by reading acp/task/sender.py) -- the
        drain loop's SECOND send (the "now running" notification for the
        queued item) hits exactly that. Without the timeout this would hang
        the test -- and a real turn -- forever instead of ever reaching the
        reinsert-and-raise path.
        """
        import acp_adapter.server as server_module
        from acp.agent.connection import AgentSideConnection

        monkeypatch.setattr(server_module, "_SESSION_UPDATE_TIMEOUT_SECONDS", 0.2)

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        def _run(*args, **kwargs):
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        loop = asyncio.get_running_loop()
        read_fd, write_fd = os.pipe()
        # Close the read end immediately: every write to write_fd now fails,
        # the real-world equivalent of the client process/pipe going away.
        os.close(read_fd)
        write_file = os.fdopen(write_fd, "wb", buffering=0)
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, write_file
        )
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
        # Never read from; only needed to satisfy AgentSideConnection's
        # StreamReader type check (listening=False skips the receive loop).
        reader = asyncio.StreamReader(limit=1024 * 1024, loop=loop)

        conn = AgentSideConnection(agent, writer, reader, listening=False)
        # A prompt that arrived (and got queued) while this turn was
        # already running.
        state.queued_prompts.append("follow-up while busy")
        try:
            with pytest.raises(Exception):
                await agent.prompt(
                    prompt=[TextContentBlock(type="text", text="hi")],
                    session_id=resp.session_id,
                )

            assert state.is_running is False
            # The queued item must still be there for a later retry, not lost.
            assert state.queued_prompts == ["follow-up while busy"]
        finally:
            # The sender's background task already died from the same
            # broken pipe being asserted on above; close() re-awaits that
            # task and would otherwise re-raise its exception during
            # cleanup. That's expected here -- the connection is known
            # broken by design -- so only swallow the transport-level
            # errors close() surfaces, not a real assertion failure.
            try:
                await conn.close()
            except (ConnectionResetError, BrokenPipeError, OSError, asyncio.TimeoutError):
                pass

    @pytest.mark.asyncio
    async def test_is_running_resets_when_provenance_send_hits_dead_connection(
        self, agent, mock_manager, monkeypatch
    ):
        """Compression-rotation sibling of the delivery-failure fix above: ``_finish_turn``
        sends a pre-drain provenance update (``_send_session_info_update`` -> ``_send``)
        BEFORE the try/finally that resets ``is_running``. That path used an unbounded
        ``_send``, so on a connection that silently hangs instead of raising (the same
        dead-sender behavior documented above), the provenance await never returns and the
        reset guard below it is never reached -- the exact bug this PR fixes, just on the
        rotation path instead of final-response delivery.

        Turn 1 dies against the closed pipe for real, killing ``MessageSender``'s background
        loop; turn 2 rotates ``agent.session_id`` so ``_finish_turn``'s FIRST write is the
        provenance update, landing on that now-dead connection's silent-hang behavior instead
        of a raise. ``_send`` must now be bounded by the same timeout for turn 2 to ever
        finish instead of hanging the test (and a real turn) forever.
        """
        import acp_adapter.server as server_module
        from acp.agent.connection import AgentSideConnection

        monkeypatch.setattr(server_module, "_SESSION_UPDATE_TIMEOUT_SECONDS", 0.2)

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.agent.session_id = "hermes-1"

        def _run_no_rotation(*args, **kwargs):
            return {"final_response": "ok", "messages": []}

        def _run_with_rotation(*args, **kwargs):
            # Simulate a mid-turn compression split: the internal head changes as a
            # side effect of THIS run, so pre_turn_hermes_id (snapshotted before the
            # executor call) differs from post_turn_hermes_id (read after it returns).
            state.agent.session_id = "hermes-2"
            # Non-empty messages: SessionManager._persist() leaves a brand-new session
            # ephemeral (no DB row at all) until it has real history, and
            # _send_session_info_update() silently no-ops -- never touching the
            # connection -- when it can't find a DB row for the session. Turn 1 never
            # created one (it also returned empty messages), so without this, turn 2's
            # provenance send would skip the real transport entirely, and this test
            # would pass regardless of whether _send() is bounded (verified: it does,
            # even with the fix reverted).
            return {
                "final_response": "ok",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }

        state.agent.run_conversation = _run_no_rotation

        loop = asyncio.get_running_loop()
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        write_file = os.fdopen(write_fd, "wb", buffering=0)
        transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, write_file)
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
        reader = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
        conn = AgentSideConnection(agent, writer, reader, listening=False)
        agent._conn = conn

        try:
            # Turn 1: the FIRST write on this connection fails for real (closed pipe),
            # killing the sender's background loop -- expected to raise.
            with pytest.raises(Exception):
                await agent.prompt(
                    prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id
                )
            assert state.is_running is False

            # Turn 2: the run itself rotates the internal head, so _finish_turn's first
            # action is the provenance send, hitting the now-dead connection's
            # hang-not-raise behavior.
            state.agent.run_conversation = _run_with_rotation
            with pytest.raises(Exception):
                await agent.prompt(
                    prompt=[TextContentBlock(type="text", text="again")], session_id=resp.session_id
                )
            assert state.is_running is False
        finally:
            try:
                await conn.close()
            except (ConnectionResetError, BrokenPipeError, OSError, asyncio.TimeoutError):
                pass

    @pytest.mark.asyncio
    async def test_queue_drains_after_recoverable_failure_but_is_not_replayed_once_run(
        self, agent, mock_manager
    ):
        """Two more facets of the same delivery-failure guarantee, covered
        as scenario blocks in one test so the drain loop can't satisfy one
        by breaking the other (e.g. "always retry the notification" would
        pass scenario 1 but replay work in scenario 2).

        Scenario 1 -- one-off failure, connection recovers: the ORIGINAL
        final-response delivery fails, but the connection has recovered by
        the time the drain loop retries the "now running" notification for
        a queued item -- ``is_running`` must still reset AND the queued
        item must fully drain (``self.prompt()`` actually runs it), not
        just survive unexecuted.

        Scenario 2 -- nested turn already ran: once ``self.prompt()`` picks
        the item up, it owns it -- a failure in THAT nested turn's own
        final-response delivery (which happens only after the nested turn
        already ran, tools included, and persisted its history) must not
        put the item back on the queue. Requeuing it would replay an
        already-executed turn, side-effecting tool calls included, a
        second time.
        """

        def _make_run(run_calls):
            def _run(*args, **kwargs):
                run_calls.append(1)
                return {"final_response": "ok", "messages": []}

            return _run

        # --- Scenario 1: one-off failure, connection recovers -> full drain
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        run_calls: list[int] = []
        state.agent.run_conversation = _make_run(run_calls)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        call_count = {"n": 0}

        async def flaky_session_update(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("connection dropped")
            return None

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock(side_effect=flaky_session_update)
        agent._conn = mock_conn

        # Simulate a prompt that arrived (and got queued) while this turn
        # was already running.
        state.queued_prompts.append("follow-up while busy")

        with pytest.raises(RuntimeError):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="hi")],
                session_id=resp.session_id,
            )

        assert state.is_running is False
        assert state.queued_prompts == []
        assert len(run_calls) == 2  # original turn + the drained follow-up

        # --- Scenario 2: nested turn already ran -> item is NOT replayed --
        resp2 = await agent.new_session(cwd=".")
        state2 = mock_manager.get_session(resp2.session_id)
        run_calls2: list[int] = []
        state2.agent.run_conversation = _make_run(run_calls2)
        state2.agent.model = "test-model"
        state2.agent.provider = "openrouter"

        # Sequence: [1] original turn's final response (ok), [2] "now
        # running" notification for the queued item (ok), [3] the NESTED
        # turn's own final-response delivery (fails).
        call_count2 = {"n": 0}

        async def flaky_session_update2(*args, **kwargs):
            call_count2["n"] += 1
            if call_count2["n"] == 3:
                raise RuntimeError("nested delivery failed")
            return None

        mock_conn2 = MagicMock(spec=acp.Client)
        mock_conn2.session_update = AsyncMock(side_effect=flaky_session_update2)
        agent._conn = mock_conn2
        state2.queued_prompts.append("follow-up while busy")

        with pytest.raises(RuntimeError):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="hi")],
                session_id=resp2.session_id,
            )

        # The nested turn actually ran -- must not be replayed.
        assert state2.queued_prompts == []
        assert len(run_calls2) == 2














# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0




    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force, **kwargs):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            focus_topic=None,
            force=True,
            defer_context_engine_notification=True,
            task_id=state.session_id,
        )
        mock_save.assert_called_once_with(state.session_id)


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        before = str(resolve_agent_cwd())
        agent._handle_slash_command("/help", state)
        assert str(resolve_agent_cwd()) == before





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}


    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
