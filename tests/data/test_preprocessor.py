import numpy as np
from PIL import Image
from olmo.data.data_formatter import DataFormatter
from olmo.data.model_preprocessor import Preprocessor, MultiModalPreprocessor
from olmo.html_utils import postprocess_prompt
from olmo.tokenizer import build_tokenizer, IMAGE_PROMPT

DUMMY_IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


def get_preprocessor():
    tokenizer = build_tokenizer("Qwen/Qwen2-7B")
    return MultiModalPreprocessor(tokenizer=tokenizer)


def _test_tokenization(messages, n_at_start=None, preprocessor=None):
    if preprocessor is None:
        preprocessor = get_preprocessor()
    if n_at_start is None:
        n = " ".join(messages).count(IMAGE_PROMPT)
    else:
        n = n_at_start
    batch = preprocessor([DUMMY_IMAGE]*n, messages)
    with_tokens = preprocessor.tokenizer.decode(batch["target_tokens"], truncate_at_eos=False)
    out = postprocess_prompt(with_tokens)
    if n_at_start is None:
        expected = "".join(messages).replace(IMAGE_PROMPT, "IMAGE[144]")
    else:
        expected = "".join(["IMAGE[144]"]*n + messages)
    assert out == expected, f"Expected \"{expected}\", but got \"{out}\""


def test_text_only():
    _test_tokenization(["What time is it?", " 3"])
    _test_tokenization(["a b", " res1 res2", " d e", " res5"])


def test_at_start():
    _test_tokenization([" question?", " answer"], 1)
    _test_tokenization([" a long question", " ans"], 2)
    _test_tokenization([" a long question", " ans", " next", " end"], 2)


def test_in_message():
    _test_tokenization([f"look at {IMAGE_PROMPT} what is it?", "answer"])
    _test_tokenization(
        [f"1: {IMAGE_PROMPT} 2: {IMAGE_PROMPT} 3: {IMAGE_PROMPT}?", "answer"],
    )
    _test_tokenization([
        f"1: {IMAGE_PROMPT} 2: {IMAGE_PROMPT} 3: {IMAGE_PROMPT}?",
        "response 1",
        f"4: {IMAGE_PROMPT}",
        "response 2",
    ])


def test_multi_message():
    pre = get_preprocessor()
    messages = [
        [" turn11", " turn12", " turn13", " turn14"],
        [" turn21", " turn22"]
    ]
    batch = pre([DUMMY_IMAGE], messages)
    out = pre.tokenizer.decode(batch["target_tokens"], truncate_at_eos=False)
    out = postprocess_prompt(out)
    assert out == "IMAGE[144] turn11 turn12 turn13 turn14 turn21 turn22"

    seg = batch["subsegment_ids"]
    tokens = batch["input_tokens"]

    seg1 = pre.tokenizer.decode(tokens[seg != seg[-1]], truncate_at_eos=False)
    assert postprocess_prompt(seg1) == "IMAGE[144] turn11 turn12 turn13 turn14"

    seg2 = pre.tokenizer.decode(tokens[seg >= seg[-1]], truncate_at_eos=False)
    assert postprocess_prompt(seg2) == "IMAGE[144] turn21 turn22"
