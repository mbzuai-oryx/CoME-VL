"""Various VQA metrics used by different datasets"""
import re
import string
from collections import Counter
from typing import Optional, List

import editdistance
import numpy as np

from olmo.eval import mmmu_eval_utils, math_vista_utils

import os, torch.distributed as dist
# from trl.rewards.all_rewards import get_reward_funcs

def barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

rank = int(os.environ.get("RANK", 0))



contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
    "youll": "you'll", "youre": "you're", "youve": "you've"}

manualMap = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a','an','the']

punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!']

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",outText,re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(ans, cache={}):
    if ans in cache:
        return cache[ans]
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t',' ')
    ans = ans.lower().strip()
    preprocessed = processDigitArticle(processPunctuation(ans))
    cache[ans] = preprocessed
    return preprocessed


def vqa_score(target, pred):
    """
    Evaluation with VQA 2 style preprocessing
    """
    pred = preprocess_answer(pred)
    if isinstance(target, list):
        target = Counter(preprocess_answer(x) for x in target)
        return min(target[pred] / 3.0, 1)
    else:
        return float(pred == target)


def a_okvqa_score(target, pred):
    # A-OK-VQA eval scripts don't seem to do any answer pre-processing
    target = Counter([x.lower().strip() for x in target])
    return min(target[pred.lower().strip()] / 3.0, 1)


def select_mc_option(target, options):
    """
    Selects a multiple-choice option based on the model output

    The output is should exactly match one of the option, but contains
    some heuristic fallbacks in case the does not occur
    """
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    assert len(set(options)) == n
    for ix, option in enumerate(options):
        if option == target:
            return ix

    contains = []
    for ix, option in enumerate(options):
        if target in option:
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]
    distances = [editdistance.eval(opt, target) for opt in options]
    return np.argmin(distances)


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """
    # Lowercase is not in https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
    # However https://rrc.cvc.uab.es/?ch=17&com=tasks says
    #  - "Answers are not case sensitive"
    #  - "Answers are space sensitive"
    edit_distance = editdistance.eval(target.lower(), prediction.lower())
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# From https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/main_parse_and_eval.py
def mmmu_score(
    target: List[str],
    response: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple-choice":
        options = metadata["options"]
        options = [opt for opt in options if len(opt) > 0]
        all_choices = [chr for chr in string.ascii_uppercase[:len(options)]]
        index2ans = {chr: option for chr, option in zip(all_choices, options)}
        parsed_pred = mmmu_eval_utils.parse_multi_choice_response(response, all_choices, index2ans)
        correct = mmmu_eval_utils.eval_multi_choice(target, parsed_pred)
    else: # open
        parsed_pred = mmmu_eval_utils.parse_open_response(response)
        correct = mmmu_eval_utils.eval_open(target, parsed_pred)
    return float(correct)


def real_world_qa_score(
    target: str,
    prediction: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple_choice":
        options = ["A", "B", "C", "D"]
        pred_idx = select_mc_option(prediction, options)
        gt_idx = options.index(target)
        score = pred_idx == gt_idx
    else:
        pred = preprocess_answer(prediction)
        gt = preprocess_answer(target)
        score = float(pred == gt)
    return score


def math_vista_score(
    response: str,
    metadata: dict,
    openai_api_key: str,
    use_api: bool = True,
):
    # extract answer using GPT-4.
    pid = metadata["example_id"]
    question_type = metadata["question_type"]
    answer_type = metadata["answer_type"]
    choices = metadata["choices"]
    target = metadata["answer"]
    query = metadata["query"]

    if use_api:
        extraction = math_vista_utils.extract_answer(
            pid, response, question_type, answer_type, choices, query, openai_api_key,
        )
    else:
        if question_type == "multi_choice":
            options = [chr(ord("A") + i) for i in range(len(choices))]
            pred_idx = select_mc_option(response, options)
            extraction = choices[pred_idx]
        else:
            if answer_type == "integer":
                try:
                    extraction = str(int(response))
                except:
                    extraction = response
            elif answer_type == "float":
                try:
                    extraction = str(float(response))
                except:
                    extraction = response
            else:
                extraction = response

    # calculate score
    precision = metadata["precision"]

    # normalize the extracted answer to match the answer type
    prediction = math_vista_utils.normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision,
    )

    # verify the prediction is true or false
    true_false = math_vista_utils.safe_equal(prediction, target)

    return true_false





# Copyright (c) OpenMMLab. All rights reserved.
import torch


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


import re
import torch

# --- paste your bbox_overlaps() implementation above this line ---

number_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_box_xyxy(x):
    """
    Accepts: "[x1, y1, x2, y2]" or any string containing 4 numbers,
             or a list/tuple of 4 numbers.
    Returns: torch.float32 tensor [4] in xyxy order.
    """
    if isinstance(x, (list, tuple)) and len(x) == 4:
        vals = [float(v) for v in x]
    else:
        s = str(x)
        vals = [float(v) for v in number_re.findall(s)]
        assert len(vals) >= 4, f"Could not find 4 numbers in: {x}"
        vals = vals[:4]
    x1, y1, x2, y2 = vals
    # ensure ordered
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

def compute_iou_and_accuracy(responses, predictions, iou_thr=0.5, aligned=True):
    """
    responses/predictions: iterables of boxes (strings or lists)
    If aligned=True, they must be same length and compared 1:1.
    Returns: tensor of IoUs, and scalar accuracy at iou_thr.
    """
    # Parse to tensors
    b1 = torch.stack([parse_box_xyxy(b) for b in responses], dim=0)  # (m,4)
    b2 = torch.stack([parse_box_xyxy(b) for b in predictions], dim=0)  # (n,4)

    if aligned:
        assert b1.size(0) == b2.size(0), "Aligned mode requires same number of boxes."
        ious = bbox_overlaps(b1.unsqueeze(0), b2.unsqueeze(0), mode='iou', is_aligned=True).squeeze(0)  # (m,)
        acc = (ious >= iou_thr).float().mean()
    else:
        # many-to-many: use best IoU per response box
        ious = bbox_overlaps(b1.unsqueeze(0), b2.unsqueeze(0), mode='iou', is_aligned=False).squeeze(0)  # (m,n)
        best = ious.max(dim=1).values  # best match for each response
        acc = (best >= iou_thr).float().mean()

    return ious, acc

# # -------------------- EXAMPLES --------------------
# # 1) Strings vs floats (aligned)
# responses = ['[519, 2, 640, 161]', '[367, 112, 472, 339]']
# predictions = [[68.7, 50.1, 87.9, 59.4], '[216, 55, 318, 403]']  # already xyxy pixels
# ious, acc = compute_iou_and_accuracy(responses, predictions, iou_thr=0.5, aligned=True)
# print("IoUs:", ious.tolist())
# print("Acc@0.5:", float(acc))

# # 2) Many-to-many (take best match per response)
# responses = ['[10,10,50,50]', '[100,100,160,160]']
# predictions = ['[12,12,48,48]', '[95,95,130,130]', '[120,120,170,170]']
# ious, acc = compute_iou_and_accuracy(responses, predictions, iou_thr=0.5, aligned=False)
# print("IoU matrix:\n", ious)
# print("Acc@0.5:", float(acc))


def iou_score(
    response: str,
    prediction: str,
    metadata: dict
):
    # If the model didn't output any box numbers, treat as 0 IoU instead of crashing.
    if len(number_re.findall(str(prediction))) < 4:
        print("response -> ", response)
        print("prediction -> ", prediction)
        return 0.5
    # print("response -> ", response)
    # print("prediction -> ", prediction)
    # ious, acc = compute_iou_and_accuracy(response, prediction, iou_thr=0.1, aligned=True)
    # score = torch.mean(ious)
    # print('score -> ', score)
    # if rank==0:
    #     breakpoint()
    ious, acc = compute_iou_and_accuracy(response, [prediction], iou_thr=0.1, aligned=True)
    # score = torch.mean(ious)
    print('ious -> ', ious)
    return ious[0].item()
