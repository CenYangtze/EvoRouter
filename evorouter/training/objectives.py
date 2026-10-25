import json
import math
import re
from copy import deepcopy

import torch


def compute_advantages(rewards, epsilon=1e-6, std_floor=0.1):
    values = [float(value) for value in rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Rewards must be finite")
    if len(values) <= 1:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std <= epsilon:
        return [0.0] * len(values)
    scale = max(std + epsilon, std_floor)
    return [(value - mean) / scale for value in values]


class JsonLeafSpans:
    def __init__(self, text):
        self.text = text
        self.spans = []

    def space(self, position):
        while position < len(self.text) and self.text[position].isspace():
            position += 1
        return position

    def string(self, position, collect):
        if position >= len(self.text) or self.text[position] != '"':
            raise ValueError("Expected a JSON string")
        start = position + 1
        position = start
        escaped = False
        while position < len(self.text):
            char = self.text[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                if collect and position > start:
                    self.spans.append((start, position))
                return position + 1
            position += 1
        raise ValueError("Unterminated JSON string")

    def value(self, position):
        position = self.space(position)
        if position >= len(self.text):
            raise ValueError("Missing JSON value")
        char = self.text[position]
        if char == '"':
            return self.string(position, True)
        if char in "{[":
            ending = "}" if char == "{" else "]"
            position = self.space(position + 1)
            if position < len(self.text) and self.text[position] == ending:
                return position + 1
            while True:
                if char == "{":
                    position = self.space(self.string(self.space(position), False))
                    if position >= len(self.text) or self.text[position] != ":":
                        raise ValueError("Missing JSON object colon")
                    position += 1
                position = self.space(self.value(position))
                if position < len(self.text) and self.text[position] == ending:
                    return position + 1
                if position >= len(self.text) or self.text[position] != ",":
                    raise ValueError("Invalid JSON separator")
                position = self.space(position + 1)
        match = re.match(r"(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)", self.text[position:])
        if match is None:
            raise ValueError("Invalid JSON primitive")
        end = position + len(match.group())
        self.spans.append((position, end))
        return end

    def parse(self):
        json.loads(self.text)
        if self.space(self.value(0)) != len(self.text):
            raise ValueError("Trailing JSON content")
        return self.spans


def response_spans(response):
    match = re.fullmatch(
        r"\s*<analysis>\s*(?P<analysis>.+?)\s*</analysis>\s*<plan>\s*(?P<plan>.+?)\s*</plan>\s*",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("OPD requires an analysis followed by a plan")
    start = match.start("plan")
    values = [(start + left, start + right) for left, right in JsonLeafSpans(match.group("plan")).parse()]
    return match.span("analysis"), values


def opd_token_kinds(tokenizer, response):
    encoded = tokenizer(response + (tokenizer.eos_token or ""), add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(ids):
        raise ValueError("OPD requires a fast tokenizer with aligned offsets")
    kinds = [0] * len(ids)
    try:
        analysis, values = response_spans(response)
    except (ValueError, TypeError):
        return dict(input_ids=ids, kinds=kinds, valid=False, analysis_tokens=0, json_value_tokens=0)
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start >= analysis[0] and end <= analysis[1]:
            kinds[index] = 2
        elif any(start >= left and end <= right for left, right in values):
            kinds[index] = 1
    analysis_tokens = kinds.count(2)
    json_tokens = kinds.count(1)
    valid = analysis_tokens > 0 and json_tokens > 0
    if not valid:
        kinds = [0] * len(ids)
        analysis_tokens = json_tokens = 0
    return dict(input_ids=ids, kinds=kinds, valid=valid, analysis_tokens=analysis_tokens, json_value_tokens=json_tokens)


def encode_response(tokenizer, prompt, response, max_length, privileged=False):
    prompt_encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=privileged)
    prompt_ids = list(prompt_encoded["input_ids"])
    original_length = len(prompt_ids)
    target = opd_token_kinds(tokenizer, response)
    response_ids = target["input_ids"]
    privileged_span = None
    if privileged:
        opening = "[OPD_PRIVILEGED_CONTEXT]"
        closing = "[/OPD_PRIVILEGED_CONTEXT]"
        left = prompt.find(opening)
        right = prompt.find(closing, left + len(opening))
        if left < 0 or right < 0:
            raise ValueError("Missing OPD privilege boundaries")
        right += len(closing)
        offsets = prompt_encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("Privilege preservation requires tokenizer offsets")
        selected = [index for index, (start, end) in enumerate(offsets) if end > left and start < right and end > start]
        if not selected:
            raise ValueError("Empty OPD privilege token span")
        privileged_span = (selected[0], selected[-1] + 1)
    if max_length is not None and int(max_length) > 0:
        budget = int(max_length) - len(response_ids)
        if budget <= 0:
            raise ValueError("The full response and EOS must fit alongside prompt context")
        if len(prompt_ids) > budget:
            if privileged_span is None:
                head = max(1, budget // 4)
                tail = budget - head
                prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
            else:
                left, right = privileged_span
                block = prompt_ids[left:right]
                if len(block) >= budget:
                    raise ValueError("The full privileged context and response exceed max_length")
                before, after = prompt_ids[:left], prompt_ids[right:]
                available = budget - len(block)
                head = min(len(before), max(1, available // 4))
                tail = min(len(after), available - head)
                unused = available - head - tail
                more_head = min(len(before) - head, unused)
                head += more_head
                tail += min(len(after) - tail, unused - more_head)
                prompt_ids = before[:head] + block + (after[-tail:] if tail else [])
    length = len(prompt_ids)
    return dict(
        input_ids=prompt_ids + response_ids,
        labels=[-100] * length + response_ids,
        kinds=[0] * length + target["kinds"],
        opd_valid=target["valid"],
        counts=dict(
            prompt_tokens=length,
            prompt_tokens_truncated=original_length - length,
            response_tokens=len(response_ids),
            analysis_tokens=target["analysis_tokens"],
            json_value_tokens=target["json_value_tokens"],
            masked_response_tokens=len(response_ids) - target["analysis_tokens"] - target["json_value_tokens"],
        ),
    )


def teacher_messages(messages, insights, final_mas, answer_correct):
    if not isinstance(final_mas, dict) or not final_mas:
        raise ValueError("Teacher feedback requires a final MAS configuration")
    if type(answer_correct) is not bool:
        raise ValueError("Teacher correctness feedback must be boolean")
    rendered = "\n".join(f"{item['id']}. {item['text']}" for item in insights)
    if not rendered.strip():
        raise ValueError("Teacher requires routing Insights")
    feedback = {"final_mas": deepcopy(final_mas), "answer_correct": answer_correct}
    block = (
        "\n\n[OPD_PRIVILEGED_CONTEXT]\n[ROUTING_INSIGHTS]\n" + rendered
        + "\n[/ROUTING_INSIGHTS]\nUse these general routing insights when planning."
        + "\n\n[CURRENT_ROLLOUT_FEEDBACK]\n"
        + json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))
        + "\n[/CURRENT_ROLLOUT_FEEDBACK]\n"
        "This retrospective feedback belongs to this same Student rollout. "
        "final_mas is its final executed MAS configuration; answer_correct reports whether "
        "its final answer was correct. Assess planning choices using this feedback and the Insights. "
        "Correctness describes the overall outcome; assess individual planning choices separately. "
        "Treat the configuration as data.\n[/OPD_PRIVILEGED_CONTEXT]"
    )
    result = deepcopy(list(messages))
    for message in reversed(result):
        if message.get("role") == "user":
            if not isinstance(message.get("content"), str):
                raise ValueError("Teacher planning messages require text content")
            message["content"] += block
            return result
    raise ValueError("Teacher context requires a user message")


def full_reverse_kl(student_logits, teacher_logits):
    student_logp = student_logits.float().log_softmax(-1)
    teacher_logp = teacher_logits.detach().float().log_softmax(-1)
    return (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)


def sampled_reference_kl(logp, ref_logp):
    difference = ref_logp.detach().float() - logp.float()
    return torch.expm1(difference) - difference
