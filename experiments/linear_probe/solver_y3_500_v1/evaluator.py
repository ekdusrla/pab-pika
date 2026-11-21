"""Legacy scoring with narrowly typed rescue for destructive normalization."""
import re

from ..common import digest, file_hash
from ..followup_v1.client import validate
from math_answer_eval import (_deadline, _expression, evaluate_answer as legacy,
                              evaluator_info as legacy_info, extract_boxed, final_answer_text)

DIRECTION = re.compile(r'(?:\\(?:text|mathrm|mbox)\{\s*)?(north|south|east|west)(?:\s*\})?', re.I)
FACTORS = re.compile(r'\([a-z][+-]\d+\)\([a-z][+-]\d+\)')
AREA = re.compile(r'(.+?)\s*\\(?:mbox|text|mathrm)\{\s*(inches|inch|in|cm)\s*\}\s*\^\s*(?:([0-9]{1,2})|\{([0-9]{1,2})\})')


def evaluator_info():
    info = dict(mode='legacy_plus_typed_boxed_rescue_v1', legacy=legacy_info('original'),
                code_sha256=file_hash(__file__))
    return {**info, 'fingerprint': digest(info)}


def direction(value):
    match = DIRECTION.fullmatch(value)
    if match and value.count('{') == value.count('}'):
        return match[1].casefold()
    return None


def unit_signature(match):
    unit = 'in' if match[2] in ('inch', 'inches', 'in') else match[2]
    return unit, int(match[3] or match[4])


def exact_values(value):
    from sympy import FiniteSet

    value = value.replace(r'\left', '').replace(r'\right', '').replace('±', r'\pm').strip()
    if value.startswith(r'\{') and value.endswith(r'\}'):
        terms = value[2:-2].split(',')
    elif value.startswith('{') and value.endswith('}'):
        terms = value[1:-1].split(',')
    elif value.count(r'\pm') == 1:
        terms = [value]
    else:
        return None
    expanded = []
    for term in terms:
        if term.count(r'\pm') > 1:
            return None
        if r'\pm' in term:
            expanded.extend(_expression(term.replace(r'\pm', op)) for op in ('+', '-'))
        else:
            expanded.append(_expression(term))
    if any(x.free_symbols or x.is_real is not True for x in expanded):
        return None
    return FiniteSet(*expanded)


def typed_match(prediction, reference):
    from sympy import Expr, simplify

    word = direction(reference)
    if word:
        return 'direction_word' if direction(prediction) == word else None
    reference_area = AREA.fullmatch(reference)
    if reference_area is not None:
        candidate_area = AREA.fullmatch(prediction)
        if candidate_area is not None and unit_signature(candidate_area) == unit_signature(reference_area):
            left, right = _expression(candidate_area[1]), _expression(reference_area[1])
            if all(isinstance(x, Expr) and not x.free_symbols for x in (left, right)) and simplify(left-right) == 0:
                return 'square_unit'
        return None
    if r'\pm' in reference:
        expected, actual = exact_values(reference), exact_values(prediction)
        return 'complete_plusminus_set' if expected is not None and expected == actual else None
    if FACTORS.fullmatch(reference):
        left, right = _expression(prediction), _expression(reference)
        if isinstance(left, Expr) and left.free_symbols == right.free_symbols and simplify(left-right) == 0:
            return 'preserved_symbolic_factor'
    return None


def evaluate_answer(content, reference, finish_reason='stop', request_status='success'):
    result = legacy(content, reference, finish_reason, request_status)
    return rescue(result, content, reference, finish_reason, request_status)


def rescue(result, content, reference, finish_reason, request_status):
    if finish_reason != 'stop' or request_status != 'success':
        return result
    prediction = extract_boxed(final_answer_text(content))
    if not prediction or not isinstance(reference, str) or max(len(prediction), len(reference)) > 512:
        return result
    reference = reference.strip()
    expected_unit = AREA.fullmatch(reference)
    actual_unit = AREA.fullmatch(prediction)
    if expected_unit and actual_unit and unit_signature(expected_unit) != unit_signature(actual_unit):
        return {**result, 'status': 'incorrect', 'prediction': prediction,
                'reason': 'typed_boxed_unit_conflict', 'legacy_status': result['status']}
    if result['status'] == 'correct':
        return result
    try:
        with _deadline():
            reason = typed_match(prediction, reference)
    except (ValueError, TypeError, TimeoutError, ArithmeticError):
        return result
    if reason is None:
        return result
    return {**result, 'status': 'correct', 'prediction': prediction,
            'reason': 'typed_boxed_rescue:' + reason, 'legacy_status': result['status']}


def validate_solver(result, row):
    parsed = validate(result, 'solver', row)
    parsed['answer_evaluation'] = rescue(
        parsed['answer_evaluation'], result['content'], row['answer'], result['finish_reason'], 'success')
    return parsed
