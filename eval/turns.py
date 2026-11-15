"""Single-line action protocol for interactive evaluation, not ETO replay."""
import re


PROTOCOL = 'single-action-line-v2'
ACTION = re.compile(r'^[ \t]*Action:[ \t]*([^\r\n]*)', re.MULTILINE)


def action_turn(text, final=False):
    """Return (accepted assistant text, action), or None until a line ends.

    EOS can terminate an action line; a token-budget cutoff cannot. Only the
    first Action line is eligible, and nothing after it enters the history.
    """
    match = ACTION.search(text)
    if match is None:
        if final:
            raise ValueError("no 'Action:' line in model output")
        return None
    if match.end() == len(text) and not final:
        return None
    action = match.group(1).strip(' \t`')
    if not action:
        raise ValueError('empty Action line')
    return text[:match.end()].rstrip(), action


def cache_length(past):
    if hasattr(past, 'get_seq_length'):
        return past.get_seq_length()
    return past[0][0].shape[-2]


def crop_cache(past, length):
    """Roll back speculative turn tokens, including a boundary-spanning token."""
    if hasattr(past, 'crop'):
        past.crop(length)
        return past
    return tuple(tuple(value[..., :length, :] for value in layer) for layer in past)
