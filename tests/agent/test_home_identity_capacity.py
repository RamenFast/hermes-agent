"""Identity-bound home retry policy.

The resident's home ACP may rotate credentials for the exact active provider/model,
but it must never substitute a configured fallback model. Long provider waits end
through the existing failed-turn seam once exact credentials are exhausted.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.chat_completion_helpers import try_activate_fallback
from agent.conversation_loop import _home_identity_capacity_policy
from agent.error_classifier import FailoverReason
from run_agent import AIAgent


def _home_agent(**overrides):
    values = {
        "session_class": "home",
        "provider": "anthropic",
        "model": "claude-opus-4-8",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_long_home_rate_limit_rotates_now_then_fails_fast_if_unrecovered():
    rotate, fail_fast = _home_identity_capacity_policy(
        _home_agent(),
        FailoverReason.rate_limit,
        {"reset_at": time.time() + 600},
    )
    assert rotate is True
    assert fail_fast is True


def test_short_home_rate_limit_keeps_normal_same_credential_retry():
    rotate, fail_fast = _home_identity_capacity_policy(
        _home_agent(),
        FailoverReason.rate_limit,
        {"reset_at": time.time() + 5},
    )
    assert rotate is False
    assert fail_fast is False


def test_workspace_rate_limit_keeps_general_retry_and_fallback_policy():
    rotate, fail_fast = _home_identity_capacity_policy(
        _home_agent(session_class="workspace"),
        FailoverReason.rate_limit,
        {"reset_at": time.time() + 600},
    )
    assert rotate is False
    assert fail_fast is False


def test_home_billing_wall_fails_fast_without_credential_wait():
    rotate, fail_fast = _home_identity_capacity_policy(
        _home_agent(),
        FailoverReason.billing,
        {},
    )
    assert rotate is False
    assert fail_fast is True


def test_forced_rate_limit_recovery_rotates_on_the_first_429():
    current = SimpleNamespace(last_status=None)
    replacement = SimpleNamespace(id="claude-code")
    pool = MagicMock()
    pool.provider = "anthropic"
    pool.current.return_value = current
    pool.mark_exhausted_and_rotate.return_value = replacement
    agent = _home_agent(
        _credential_pool=pool,
        api_key="oauth-token",
        _swap_credential=MagicMock(),
    )

    recovered, retried_same = recover_with_credential_pool(
        agent,
        status_code=429,
        has_retried_429=False,
        classified_reason=FailoverReason.rate_limit,
        error_context={"reset_at": time.time() + 600},
        force_rotate_rate_limit=True,
    )

    assert recovered is True
    assert retried_same is False
    pool.mark_exhausted_and_rotate.assert_called_once()
    agent._swap_credential.assert_called_once_with(replacement)


def test_home_session_refuses_configured_alternate_model_fallback():
    agent = _home_agent(
        _fallback_chain=[{"provider": "openrouter", "model": "z-ai/glm-5.2"}],
        _fallback_index=0,
    )

    assert try_activate_fallback(agent, reason=FailoverReason.rate_limit) is False
    assert agent._fallback_index == 0
    assert not hasattr(agent, "_rate_limited_until")


def test_home_conversation_returns_failed_turn_without_sleep_or_fallback():
    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "600"})

        def __str__(self):
            return "Error code: 429 - account rate limit exceeded"

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="oauth-token",
            base_url="https://api.anthropic.com",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    pool = MagicMock()
    pool.provider = "anthropic"
    pool.current.return_value = SimpleNamespace(last_status=None)
    pool.mark_exhausted_and_rotate.return_value = None

    agent.client = MagicMock()
    agent.platform = "acp"
    agent.session_class = "home"
    agent.provider = "anthropic"
    agent.model = "claude-opus-4-8"
    agent._credential_pool = pool
    agent._fallback_chain = [{"provider": "openrouter", "model": "z-ai/glm-5.2"}]
    agent._fallback_index = 0
    agent._interruptible_api_call = MagicMock(side_effect=RateLimitError())
    agent._persist_session = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None

    with patch("agent.conversation_loop.time.sleep") as sleep:
        result = agent.run_conversation("one short line back")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["failure_reason"] == FailoverReason.rate_limit.value
    assert agent._interruptible_api_call.call_count == 1
    assert agent._fallback_index == 0
    pool.mark_exhausted_and_rotate.assert_called_once()
    sleep.assert_not_called()
