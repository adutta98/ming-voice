# Unit test for the deterministic Hermes routing gate (0.3.17). Stdlib only:
# tools.py imports nothing heavy at module level. Run: python3 dev/test_hermes_gate.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import hermes_request_for, parse_tool_calls  # noqa: E402

# (transcript, expected task or None) — the None rows must fall through to normal model routing
MATCH = [
    ("Ask Hermes to give me my morning brief.", "give me my morning brief."),
    ("ask hermes to check the logs", "check the logs"),
    ("ask hermes what the traffic is like to work", "what the traffic is like to work"),
    ("Can you ask Hermes to look at the server?", "look at the server?"),
    ("please also ask Hermès to water the plants", "water the plants"),
    ("and ask hermes for my calendar tomorrow", "my calendar tomorrow"),
    ("Ask Hermes about the home to devscript commute", "the home to devscript commute"),
    ("hey, ask Hermes to tell me a joke", "tell me a joke"),
    # name + request verb not at the start: still a Hermes request, task = the whole transcript
    ("Give me my morning brief, ask Hermes to do it", "give me my morning brief, ask Hermes to do it"),
]
NOMATCH = [
    "",
    "ask hermes",                                    # nothing to hand over
    "what is hermes",                                # no request verb
    "Hermes is great, isn't it?",                    # stray mention
    "tell me about the weather",                     # no name
    "did Hermes finish the task I gave it yesterday", # status question, not a request
    "a",
]

fails = 0
for text, want in MATCH:
    got = hermes_request_for(text)
    if got != want:
        fails += 1
        print(f"MATCH FAIL {text!r}\n  want {want!r}\n  got  {got!r}")

for text in NOMATCH:
    got = hermes_request_for(text)
    if got is not None:
        fails += 1
        print(f"NOMATCH FAIL {text!r} -> {got!r}")

# The engine synthesizes this exact text after the ASSISTANT turn; the round then decodes
# from inside it and parse_tool_calls (as in Thinker.stream) must see the call + task.
for text, want in MATCH:
    task = hermes_request_for(text)
    call_text = ("<tool_call>ask_hermes\n<arg_key>task</arg_key>\n<arg_value>"
                 + task + "\n</arg_value>")
    calls = parse_tool_calls(call_text)
    if calls != [("ask_hermes", {"task": want})]:
        fails += 1
        print(f"PARSE FAIL {text!r} -> {calls!r}")

print("FAIL" if fails else f"PASS ({len(MATCH)} match, {len(NOMATCH)} no-match, parse ok)")
sys.exit(1 if fails else 0)
