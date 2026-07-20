"""Tests for the ACP ``home`` / ``workspace`` session-class register (V2-0b).

The register decides how a fresh ACP lane boots: ``workspace`` (default) keeps
the coding-agent posture (operating brief + AGENTS.md landing + git-status
snapshot) — right for editors — while ``home`` reproduces the ``hermes chat``
resident boot (SOUL identity, no workspace scaffolding). See
``agent/coding_context.py`` §"Session classes" and
``JOT-BATCH-3-ADJUDICATED.md`` §"V2-0b · The register mechanism".

Covers, per the V2-0b validation contract:
  1. class param parsing (home / workspace / absent → default)
  2. persistence round-trip (survives a fresh SessionManager reading the same DB)
  3. the home boot does NOT assemble the coding operating brief while the
     workspace boot does (assert on the real system-prompt seam)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter import server as acp_server
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState
from agent.coding_context import (
    HOME_SESSION_CLASS,
    WORKSPACE_SESSION_CLASS,
    is_home_session_class,
    normalize_session_class,
)
from agent.system_prompt import build_system_prompt_parts
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# class parsing / normalization
# ---------------------------------------------------------------------------


class TestNormalizeSessionClass:
    def test_home_is_home(self):
        assert normalize_session_class("home") == HOME_SESSION_CLASS

    def test_home_is_case_insensitive(self):
        assert normalize_session_class("HOME") == HOME_SESSION_CLASS
        assert normalize_session_class("  Home ") == HOME_SESSION_CLASS

    def test_workspace_is_workspace(self):
        assert normalize_session_class("workspace") == WORKSPACE_SESSION_CLASS

    def test_absent_defaults_to_workspace(self):
        # Back-compat: a lane that didn't ask keeps the coding-agent boot.
        assert normalize_session_class(None) == WORKSPACE_SESSION_CLASS
        assert normalize_session_class("") == WORKSPACE_SESSION_CLASS

    def test_unknown_defaults_to_workspace(self):
        assert normalize_session_class("garbage") == WORKSPACE_SESSION_CLASS

    def test_is_home_predicate(self):
        assert is_home_session_class("home") is True
        assert is_home_session_class("workspace") is False
        assert is_home_session_class(None) is False


class TestExtractSessionClassFromMeta:
    """The acp router flattens a request's ``_meta`` dict into handler kwargs
    (``params.update(meta)``), so ``_meta: {"nexus": {"class": "home"}}`` reaches
    ``new_session`` as ``kwargs["nexus"] == {"class": "home"}``."""

    def test_nested_nexus_class(self):
        assert acp_server._extract_session_class({"nexus": {"class": "home"}}) == "home"

    def test_nested_nexus_camel_case(self):
        assert acp_server._extract_session_class({"nexus": {"sessionClass": "home"}}) == "home"

    def test_nested_nexus_snake_case(self):
        assert acp_server._extract_session_class({"nexus": {"session_class": "home"}}) == "home"

    def test_top_level_flattened_class(self):
        assert acp_server._extract_session_class({"class": "home"}) == "home"

    def test_explicit_workspace(self):
        assert acp_server._extract_session_class({"nexus": {"class": "workspace"}}) == "workspace"

    def test_absent_returns_none(self):
        # None (not "workspace") so load/resume can distinguish "unspecified"
        # (keep persisted class) from "explicitly workspace".
        assert acp_server._extract_session_class({}) is None
        assert acp_server._extract_session_class({"nexus": {}}) is None
        assert acp_server._extract_session_class({"cwd": "/tmp"}) is None

    def test_unknown_value_normalizes_to_workspace(self):
        assert acp_server._extract_session_class({"nexus": {"class": "bogus"}}) == "workspace"


class TestSessionClassMeta:
    def test_builds_nexus_payload(self):
        assert acp_server._session_class_meta("home") == {"nexus": {"class": "home"}}
        assert acp_server._session_class_meta("workspace") == {"nexus": {"class": "workspace"}}

    def test_normalizes(self):
        assert acp_server._session_class_meta("HOME") == {"nexus": {"class": "home"}}

    def test_merge_keeps_both_provenance_and_class(self):
        merged = acp_server._merge_session_meta(
            {"hermes": {"sessionProvenance": {"x": 1}}},
            {"nexus": {"class": "home"}},
        )
        assert merged == {
            "hermes": {"sessionProvenance": {"x": 1}},
            "nexus": {"class": "home"},
        }

    def test_merge_drops_none_and_empties_to_none(self):
        assert acp_server._merge_session_meta(None, None) is None
        assert acp_server._merge_session_meta({}, None) is None


# ---------------------------------------------------------------------------
# SessionManager: class default, create, persistence round-trip
# ---------------------------------------------------------------------------


def _stub_agent_factory():
    # A minimal stub. _make_agent stamps ``session_class`` onto it, so we don't
    # need a real AIAgent to exercise the manager's state plumbing.
    return SimpleNamespace(model="stub-model", provider="", base_url="", api_mode="")


@pytest.fixture()
def manager():
    return SessionManager(agent_factory=_stub_agent_factory)


class TestSessionManagerClass:
    def test_default_is_workspace(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert state.session_class == WORKSPACE_SESSION_CLASS

    def test_create_home(self, manager):
        state = manager.create_session(cwd="/tmp/work", session_class="home")
        assert state.session_class == HOME_SESSION_CLASS

    def test_create_unknown_falls_back_to_workspace(self, manager):
        state = manager.create_session(cwd="/tmp/work", session_class="bogus")
        assert state.session_class == WORKSPACE_SESSION_CLASS

    def test_factory_stub_carries_class(self, manager):
        state = manager.create_session(cwd="/tmp/work", session_class="home")
        # _make_agent must stamp the register even on a stub factory so the
        # system-prompt seam sees it.
        assert state.agent.session_class == HOME_SESSION_CLASS

    def test_fork_inherits_class(self, manager):
        home = manager.create_session(cwd="/tmp/work", session_class="home")
        home.history.append({"role": "user", "content": "hi"})
        manager.save_session(home.session_id)
        forked = manager.fork_session(home.session_id, cwd="/tmp/work")
        assert forked is not None
        assert forked.session_class == HOME_SESSION_CLASS


class TestSessionClassPersistence:
    def test_home_survives_restart(self, tmp_path):
        """A ``home`` session persisted to the DB is restored as ``home`` by a
        brand-new SessionManager (simulating a process restart)."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)

        mgr1 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        state = mgr1.create_session(cwd="/tmp/work", session_class="home")
        sid = state.session_id
        # Give it a message so the restore path (which requires message_count > 0
        # to list, though get_session works regardless) has content and the
        # non-owned replace_messages path runs.
        state.history.append({"role": "user", "content": "hi"})
        mgr1.save_session(sid)

        # The register is stored in model_config JSON.
        row = db.get_session(sid)
        meta = json.loads(row["model_config"])
        assert meta.get("session_class") == "home"

        # A fresh manager over the SAME db restores it as home.
        mgr2 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        restored = mgr2.get_session(sid)
        assert restored is not None
        assert restored.session_class == HOME_SESSION_CLASS
        assert restored.agent.session_class == HOME_SESSION_CLASS

    def test_workspace_not_written_to_meta(self, tmp_path):
        """Default workspace sessions keep the model_config JSON free of a
        session_class key (byte-stable with pre-register rows)."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        mgr = SessionManager(agent_factory=_stub_agent_factory, db=db)
        state = mgr.create_session(cwd="/tmp/work")  # default workspace
        row = db.get_session(state.session_id)
        meta = json.loads(row["model_config"])
        assert "session_class" not in meta

    def test_workspace_restores_as_workspace(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        mgr1 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        state = mgr1.create_session(cwd="/tmp/work")
        state.history.append({"role": "user", "content": "hi"})
        mgr1.save_session(state.session_id)

        mgr2 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        restored = mgr2.get_session(state.session_id)
        assert restored is not None
        assert restored.session_class == WORKSPACE_SESSION_CLASS


class TestSetSessionClass:
    def test_change_workspace_to_home_preserves_history(self, manager):
        state = manager.create_session(cwd="/tmp/work")  # workspace
        state.history.extend(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        manager.save_session(state.session_id)

        updated = manager.set_session_class(state.session_id, "home")
        assert updated is not None
        assert updated.session_class == HOME_SESSION_CLASS
        # History is preserved across the agent rebuild.
        assert len(updated.history) == 2
        assert updated.agent.session_class == HOME_SESSION_CLASS

    def test_noop_when_class_matches(self, manager):
        state = manager.create_session(cwd="/tmp/work", session_class="home")
        original_agent = state.agent
        updated = manager.set_session_class(state.session_id, "home")
        assert updated is not None
        # No rebuild when the class already matches — same agent object.
        assert updated.agent is original_agent

    def test_missing_session_returns_none(self, manager):
        assert manager.set_session_class("does-not-exist", "home") is None


# ---------------------------------------------------------------------------
# The register CAUSES a different boot: home omits the coding operating brief.
# This exercises the real system-prompt seam (agent/system_prompt.py →
# coding_system_blocks), the "change the cause, not the strings" contract.
# ---------------------------------------------------------------------------


def _make_prompt_agent(session_class, **overrides):
    base = dict(
        load_soul_identity=(session_class == "home"),
        skip_context_files=(session_class == "home"),
        session_class=session_class,
        valid_tool_names=["read_file", "terminal"],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _parallel_tool_call_guidance=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _user_profile_enabled=False,
        _memory_enabled=False,
        model="claude-fable-5",
        provider="",
        platform="acp",
        pass_session_id=False,
        session_id="sess-1",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value="SOUL-IDENTITY-MARKER"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _init_code_repo(path):
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestRegisterCausesDifferentBoot:
    def test_workspace_includes_coding_brief(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_prompt_agent("workspace")
        stable = _stable_prompt(agent)
        # The coding-agent operating brief IS present for a workspace lane.
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_home_omits_coding_brief(self, monkeypatch, tmp_path):
        # Even sitting in a real code workspace, a home lane must NOT boot the
        # coding-agent posture — the register overrides cwd detection.
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_prompt_agent("home")
        stable = _stable_prompt(agent)
        assert "coding agent" not in stable
        # No git/workspace snapshot preamble either (the "Landed clean … on
        # <branch>" cause).
        assert "Workspace" not in stable
        # The resident identity (SOUL) still loads — it's her at home, not a
        # stripped coding agent.
        assert "SOUL-IDENTITY-MARKER" in stable

    def test_home_forces_general_even_when_coding_mode_on(self, monkeypatch, tmp_path):
        # config agent.coding_context: on forces the coding posture everywhere —
        # but an explicit home register still overrides it (deliberate identity
        # choice wins over the force-flag).
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_prompt_agent("home")
        with patch("agent.coding_context._coding_mode", return_value="on"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable


# ---------------------------------------------------------------------------
# Server-level: the register reaches new/load/resume via _meta and is reported
# back in the response's field_meta (_meta) so the client can render the mark.
# ---------------------------------------------------------------------------


@pytest.fixture()
def acp_agent():
    mgr = SessionManager(agent_factory=_stub_agent_factory)
    return HermesACPAgent(session_manager=mgr)


class TestServerRegisterWiring:
    @pytest.mark.asyncio
    async def test_new_session_home_via_meta(self, acp_agent):
        # The router flattens _meta into kwargs; simulate the phone's request.
        resp = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        state = acp_agent.session_manager.get_session(resp.session_id)
        assert state.session_class == "home"
        # Reported back in field_meta so the lane header can render home 🍯.
        assert resp.field_meta is not None
        assert resp.field_meta.get("nexus") == {"class": "home"}

    @pytest.mark.asyncio
    async def test_new_session_defaults_workspace(self, acp_agent):
        resp = await acp_agent.new_session(cwd="/tmp/work")
        state = acp_agent.session_manager.get_session(resp.session_id)
        assert state.session_class == "workspace"
        assert resp.field_meta.get("nexus") == {"class": "workspace"}

    @pytest.mark.asyncio
    async def test_load_session_reports_persisted_class(self, acp_agent):
        # Create a home session, then load it (no explicit class) — the load
        # response should still report home from the persisted state.
        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id
        resp = await acp_agent.load_session(cwd="/tmp/work", session_id=sid)
        assert resp is not None
        assert resp.field_meta.get("nexus") == {"class": "home"}

    @pytest.mark.asyncio
    async def test_load_session_honors_explicit_class_change(self, acp_agent):
        # A workspace lane re-opened explicitly as home flips register + reports it.
        created = await acp_agent.new_session(cwd="/tmp/work")
        sid = created.session_id
        assert acp_agent.session_manager.get_session(sid).session_class == "workspace"
        resp = await acp_agent.load_session(
            cwd="/tmp/work", session_id=sid, nexus={"class": "home"}
        )
        assert resp is not None
        assert resp.field_meta.get("nexus") == {"class": "home"}
        assert acp_agent.session_manager.get_session(sid).session_class == "home"

    @pytest.mark.asyncio
    async def test_provenance_and_class_coexist_in_meta(self, acp_agent):
        resp = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        # The nexus register rides alongside the existing hermes provenance meta,
        # never replacing it.
        assert "nexus" in resp.field_meta
        # sessionProvenance is best-effort (may be None if the DB row isn't
        # readable in the test env) but if present it must not have been clobbered.
        if "hermes" in resp.field_meta:
            assert "sessionProvenance" in resp.field_meta["hermes"]

    @pytest.mark.asyncio
    async def test_session_info_update_carries_register(self, acp_agent):
        # The register must stay visible on the post-turn session_info_update
        # notification, not just at new/load/resume — so the client's read of
        # home vs workspace never goes stale.
        from unittest.mock import AsyncMock

        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id

        conn = AsyncMock()
        acp_agent._conn = conn
        await acp_agent._send_session_info_update(sid)

        assert conn.session_update.await_count == 1
        update = conn.session_update.await_args.kwargs["update"]
        assert update.field_meta is not None
        assert update.field_meta.get("nexus") == {"class": "home"}

    def test_session_class_meta_for_unknown_session_is_none(self, acp_agent):
        # A stray update for an evicted/unknown session degrades gracefully.
        assert acp_agent._session_class_meta_for("no-such-session") is None

    @pytest.mark.asyncio
    async def test_model_switch_preserves_home_register(self, acp_agent):
        # A model switch rebuilds the agent — the home register must survive,
        # not silently revert to workspace (set_session_model path).
        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id
        with patch.object(
            acp_agent, "_resolve_model_selection", return_value=("openrouter", "some-model")
        ):
            await acp_agent.set_session_model(model_id="some-model", session_id=sid)
        state = acp_agent.session_manager.get_session(sid)
        assert state.session_class == "home"
        assert state.agent.session_class == "home"

    @pytest.mark.asyncio
    async def test_cmd_model_preserves_home_register(self, acp_agent):
        # The /model slash command rebuilds the agent too — same invariant.
        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id
        state = acp_agent.session_manager.get_session(sid)
        with patch.object(
            acp_agent, "_resolve_model_selection", return_value=("openrouter", "another-model")
        ):
            acp_agent._cmd_model("another-model", state)
        assert state.session_class == "home"
        assert state.agent.session_class == "home"

    @pytest.mark.asyncio
    async def test_fork_reports_inherited_home_class(self, acp_agent):
        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id
        # Fork needs history to copy; give it a turn.
        acp_agent.session_manager.get_session(sid).history.append(
            {"role": "user", "content": "hi"}
        )
        resp = await acp_agent.fork_session(cwd="/tmp/work", session_id=sid)
        assert resp.field_meta is not None
        assert resp.field_meta.get("nexus") == {"class": "home"}

    @pytest.mark.asyncio
    async def test_list_sessions_carries_register(self, acp_agent):
        # The session picker reads the register off each listed SessionInfo so
        # it can render the mark without a per-session load.
        created = await acp_agent.new_session(cwd="/tmp/work", nexus={"class": "home"})
        sid = created.session_id
        acp_agent.session_manager.get_session(sid).history.append(
            {"role": "user", "content": "hi"}
        )
        acp_agent.session_manager.save_session(sid)
        resp = await acp_agent.list_sessions()
        listed = {s.session_id: s for s in resp.sessions}
        assert sid in listed
        assert listed[sid].field_meta == {"nexus": {"class": "home"}}


class TestListSessionsRegister:
    """SessionManager.list_sessions carries the register for both in-memory and
    DB-restored rows so the picker mark is always available."""

    def test_in_memory_row_carries_class(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        mgr = SessionManager(agent_factory=_stub_agent_factory, db=db)
        state = mgr.create_session(cwd="/tmp/work", session_class="home")
        state.history.append({"role": "user", "content": "hi"})
        infos = mgr.list_sessions()
        row = next(i for i in infos if i["session_id"] == state.session_id)
        assert row["session_class"] == "home"

    def test_db_only_row_carries_class(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        mgr1 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        state = mgr1.create_session(cwd="/tmp/work", session_class="home")
        state.history.append({"role": "user", "content": "hi"})
        mgr1.save_session(state.session_id)
        # A fresh manager lists it from the DB (not in memory).
        mgr2 = SessionManager(agent_factory=_stub_agent_factory, db=db)
        infos = mgr2.list_sessions()
        row = next(i for i in infos if i["session_id"] == state.session_id)
        assert row["session_class"] == "home"



