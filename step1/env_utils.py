"""Shared ALFWorld environment utilities for Step 1A.

Used by trajectory generation (expert) and the evaluator (rollout) so both
see identical task_description / initial_observation extraction and the same
observation post-processing (mirrors upstream eval `process_ob` + intro strip).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


def process_ob(ob: str) -> str:
    """Upstream evaluator convention: strip 'You arrive at loc ...' prefix."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def strip_intro_text(obs: str) -> str:
    """Evaluator convention: obs = '\\n'.join(obs.split('\\n\\n')[1:])."""
    return "\n".join(obs.split("\n\n")[1:])


def split_task_and_initial_observation(obs_stripped: str) -> Tuple[str, str]:
    """task_description = the single 'Your task is to:' line (evaluator goal field);
    initial_observation = the remaining lines in original order."""
    lines = obs_stripped.split("\n")
    task_lines = [ln for ln in lines if ln.strip().startswith("Your task is to:")]
    if len(task_lines) != 1:
        raise ValueError(
            f"expected exactly one 'Your task is to:' line, found {len(task_lines)} in obs: {obs_stripped[:200]!r}"
        )
    task_description = task_lines[0].strip()
    initial_lines = [ln for ln in lines if ln is not task_lines[0]]
    initial_observation = "\n".join(initial_lines).strip()
    return task_description, initial_observation


def parse_action(llm_output: str) -> str:
    """Upstream evaluator parser: first 'Action:' match (DOTALL), raises if absent.

    Byte-identical to UPSTREAM_ACTION_RE.findall(llm_output)[0] — no extra
    stripping — so smoke parser comparisons are exact.
    """
    matches = re.findall(re.compile(r"Action:\s?(.*)", re.DOTALL), llm_output)
    if not matches:
        raise ValueError("no 'Action:' in llm_output")
    return matches[0]


def thought_for_action(action: str) -> str:
    """Deterministic thought template derived from the expert action verb.

    The handcoded expert provides actions only; Thought text is a fixed
    function of the action string (no extra information source), recorded in
    trajectory artifacts for traceability.
    """
    a = action.strip()
    low = a.lower()
    if low.startswith("go to "):
        return f"I should look for what I need; I will go to the {a[6:]} first."
    if low.startswith("take "):
        return f"I found what I need; I will {a}."
    if low.startswith("put "):
        return f"The object is ready; I will {a} to progress toward the goal."
    if low.startswith("open "):
        return f"I need to see inside; I will {a}."
    if low.startswith("close "):
        return f"I am done looking inside; I will {a}."
    if low.startswith("heat "):
        return f"This object must be heated; I will {a}."
    if low.startswith("cool "):
        return f"This object must be cooled; I will {a}."
    if low.startswith("clean "):
        return f"This object must be cleaned; I will {a}."
    if low.startswith("use "):
        return f"I will {a} to finish the task."
    if low.startswith("toggle "):
        return f"I will {a} to finish the task."
    if low == "look":
        return "I will look around carefully."
    return "I will take the next reasonable action toward the goal."


def build_receiver_messages(
    task_description: str,
    initial_observation: str,
    actions: List[str],
    observations: List[str],
) -> List[Dict[str, str]]:
    """messages for a full trajectory:
    system + user1 = instruction + task + initial obs; then
    [assistant action_k, user obs_k] pairs."""
    from step1.serialization import (
        receiver_assistant_content,
        receiver_initial_messages,
        receiver_step_user_content,
    )

    messages = receiver_initial_messages(task_description, initial_observation)
    for k, action in enumerate(actions):
        messages.append({
            "role": "assistant",
            "content": receiver_assistant_content(thought_for_action(action), action),
        })
        if k < len(observations):
            messages.append({
                "role": "user",
                "content": receiver_step_user_content(observations[k]),
            })
    return messages


def make_plain_tw_env(game_file_abs: str):
    """Plain TextWorld env for evaluator rollouts (no expert plan overhead).

    Still wraps with [AlfredDemangler, AlfredInfos] so observations use the
    same demangled entity names ("drawer 1") as training trajectories —
    identical to the name space the evaluator sees upstream.
    """
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"],
    )
    inner = textworld.start(game_file_abs, request_infos=request_infos)
    env = AlfredInfos(AlfredDemangler(shuffle=False, env=inner))
    env.load(game_file_abs)
    return env


def make_tw_env(game_file_abs: str):
    """Create an AlfredExpert-wrapped TextWorld env for one game file.

    Mirrors AlfredTWEnv.init_env's gym wrapper chain
    [AlfredDemangler, AlfredInfos, AlfredExpert] so that entity names are
    demangled to the clean ALFRED naming ("drawer 1") that both the handcoded
    expert and the paper's transcripts rely on, instead of the raw
    coordinate-encoded names stored in game.tw-pddl.
    """
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
        extras=["gamefile"],
    )
    inner = textworld.start(game_file_abs, request_infos=request_infos)
    env = AlfredExpert(
        AlfredInfos(AlfredDemangler(shuffle=False, env=inner)),
        expert_type=AlfredExpertType.HANDCODED,
    )
    # Wrapper.load cascades inward; textworld.start already loaded the inner
    # env, but the wrappers above were added afterwards, so run the chain's
    # load explicitly to apply demangling / gamefile / expert init.
    env.load(game_file_abs)
    return env


def _state_get(state, key: str, default=None):
    try:
        return state[key]
    except (KeyError, IndexError):
        return default


def tw_state_fields(state) -> Dict:
    """Extract plain fields from a TextWorld state dict-like object."""
    return {
        "feedback": state["feedback"],
        "won": bool(state["won"]),
        "admissible_commands": list(state["admissible_commands"]),
        "expert_plan": list(_state_get(state, "extra.expert_plan", []) or []),
    }
