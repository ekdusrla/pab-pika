"""Original Qwen extraction/normalization + math_verify evaluation by default.

An optional conservative evaluator remains available for explicitly named runs.
Neither mode calls a model; evaluator identity is independent of generation.
"""

import re
import signal
from contextlib import contextmanager


def final_answer_text(content):
    text = content or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    return text


def extract_boxed(text):
    """Extract the LAST boxed/fbox group, including nested braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        # Escaped braces are notation, not group delimiters.
        if text[i] in "{}" and (i == 0 or text[i - 1] != "\\"):
            depth += 1 if text[i] == "{" else -1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize(text):
    text = text.strip().replace("$", "")
    for command in (r"\left", r"\right", r"\,", r"\!", r"\;", r"\quad", r"\qquad"):
        text = text.replace(command, "")
    return re.sub(r"\s+", "", text).replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")


def _latex(text):
    def group(pos):
        if pos >= len(text) or text[pos] != "{":
            raise ValueError("LaTeX command requires a braced argument")
        depth = 1
        end = pos + 1
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError("Unbalanced braces")
        return text[pos + 1:end - 1], end

    result, i = [], 0
    while i < len(text):
        if text.startswith(r"\frac", i):
            a, j = group(i + 5)
            b, i = group(j)
            result.append(f"(({_latex(a)})/({_latex(b)}))")
        elif text.startswith(r"\sqrt", i):
            a, i = group(i + 5)
            result.append(f"sqrt({_latex(a)})")
        elif text.startswith(r"\pi", i):
            result.append("pi")
            i += 3
        elif text.startswith(r"\cdot", i):
            result.append("*")
            i += 5
        elif text.startswith(r"\times", i):
            result.append("*")
            i += 6
        elif text[i] == "\\":
            raise ValueError("Unsupported LaTeX command")
        else:
            result.append({"{": "(", "}": ")", "−": "-"}.get(text[i], text[i]))
            i += 1
    return "".join(result)


def _expression(text):
    import sympy as sp

    text = normalize(text)
    # Mixed-number notation needs a distinct grammar; do not treat it as a product.
    if re.search(r"\d\\frac", text):
        raise ValueError("Mixed numbers unsupported")
    text = _latex(text)
    tokens = re.findall(r"sqrt|pi|\d+(?:\.\d+)?|[a-zA-Z]|[()+*/^,\-]", text)
    if "".join(tokens) != text or not tokens or len(tokens) > 256:
        raise ValueError("Unsupported or oversized expression")
    pos = 0

    def expression(min_binding=0):
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Missing operand")
        token = tokens[pos]
        pos += 1
        if token in ("+", "-"):
            left = expression(25)
            if token == "-":
                left = -left
        elif token == "(":
            values = [expression()]
            while pos < len(tokens) and tokens[pos] == ",":
                pos += 1
                values.append(expression())
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Unbalanced parentheses")
            pos += 1
            left = values[0] if len(values) == 1 else sp.Tuple(*values)
        elif token == "sqrt":
            if pos >= len(tokens) or tokens[pos] != "(":
                raise ValueError("sqrt requires parentheses")
            left = sp.sqrt(expression(40))
        elif token == "pi":
            left = sp.pi
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            left = sp.Rational(token)
        elif re.fullmatch(r"[a-zA-Z]", token):
            left = sp.Symbol(token)
        else:
            raise ValueError("Unsupported operand")
        while pos < len(tokens):
            token = tokens[pos]
            if token in (",", ")"):
                break
            implicit = token not in ("+", "-", "*", "/", "^")
            op = "*" if implicit else token
            binding = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 30}[op]
            if binding < min_binding:
                break
            if not implicit:
                pos += 1
            right = expression(binding if op == "^" else binding + 1)
            if isinstance(left, sp.Tuple) or isinstance(right, sp.Tuple):
                raise ValueError("Tuple arithmetic unsupported")
            if op == "^":
                if right.free_symbols or right.is_real is not True or abs(right) > 100:
                    raise ValueError("Unsupported exponent")
                left = left ** right
            elif op == "+":
                left += right
            elif op == "-":
                left -= right
            elif op == "*":
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    result = expression()
    if pos != len(tokens):
        raise ValueError("Unexpected trailing tokens")
    if result.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError("Nonfinite expression")
    return result


@contextmanager
def _deadline(seconds=3):
    # Called synchronously by the CLI's main thread, on Linux. Restore handlers.
    def expired(*_):
        raise TimeoutError("Math evaluation timed out")
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def evaluate_conservative(content, answer, finish_reason="stop", request_status="success"):
    prediction = extract_boxed(final_answer_text(content))
    result = {"status": "undetermined", "prediction": prediction, "reference": answer}
    if request_status != "success":
        return {**result, "reason": "request_failed"}
    if finish_reason != "stop":
        return {**result, "reason": "truncated_or_abnormal_finish"}
    if prediction is None:
        return {**result, "reason": "no_complete_final_boxed_answer"}
    if not isinstance(answer, str) or not answer.strip():
        return {**result, "reason": "missing_reference"}
    reference = extract_boxed(answer) or answer
    if not prediction or max(len(prediction), len(reference)) > 2000:
        return {**result, "reason": "empty_or_oversized_answer"}
    if normalize(prediction) == normalize(reference):
        return {**result, "status": "correct", "reason": "normalized_exact_match"}
    try:
        with _deadline():
            import sympy as sp

            lhs, rhs = _expression(prediction), _expression(reference)

            def compare(a, b):
                if isinstance(a, sp.Tuple) or isinstance(b, sp.Tuple):
                    if not isinstance(a, sp.Tuple) or not isinstance(b, sp.Tuple):
                        return None
                    if len(a) != len(b):
                        return False
                    outcomes = [compare(x, y) for x, y in zip(a, b)]
                    return False if False in outcomes else (True if all(x is True for x in outcomes) else None)
                diff = sp.simplify(a - b)
                if diff == 0:
                    return True
                return diff.equals(0)

            equivalent = compare(lhs, rhs)
        if equivalent is True:
            return {**result, "status": "correct", "reason": "symbolic_equivalence"}
        if equivalent is False:
            return {**result, "status": "incorrect", "reason": "symbolic_inequivalence"}
        return {**result, "reason": "equivalence_unresolved"}
    except Exception as exc:
        return {**result, "reason": "unsupported_or_unresolved", "error_type": type(exc).__name__}


_original_parser = None


def original_parser():
    global _original_parser
    if _original_parser is None:
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[2] / "utils/qwen_math_parser.py"
        spec = importlib.util.spec_from_file_location("pab_original_math_parser", path)
        module = importlib.util.module_from_spec(spec)
        # Load the exact original module without running utils/__init__.py or dotenv.
        spec.loader.exec_module(module)
        _original_parser = module
    return _original_parser


def evaluator_info(mode="original"):
    from pathlib import Path
    from experiment_state import digest, file_hash, versions
    info = {"mode": mode, "code_sha256": file_hash(__file__)}
    if mode == "original":
        import math_verify
        original_parser()  # Fail before model calls if dependencies are missing.
        info["parser_sha256"] = file_hash(Path(__file__).resolve().parents[2] / "utils/qwen_math_parser.py")
        info["packages"] = {k: v for k, v in versions().items() if k in ("math-verify", "sympy")}
        from importlib.metadata import version
        for name in ("latex2sympy2_extended", "antlr4-python3-runtime", "word2number", "regex"):
            info["packages"][name] = version(name)
    return {**info, "fingerprint": digest(info)}


def evaluate_answer(content, answer, finish_reason="stop", request_status="success", mode="original"):
    if mode == "conservative":
        return evaluate_conservative(content, answer, finish_reason, request_status)
    result = {"status": "undetermined", "prediction": None, "reference": answer,
              "finish_reason": finish_reason}
    if request_status != "success":
        return {**result, "reason": "request_failed"}
    if not isinstance(answer, str) or not answer.strip():
        return {**result, "reason": "missing_reference"}
    try:
        import math_verify
        with _deadline(5):
            parser = original_parser()
            # Exact MathDatasetLoader.eval extraction/normalization and verifier order.
            prediction = parser.strip_string(parser.extract_answer(content or "", "math"))
            result["prediction"] = prediction
            if not prediction:
                return {**result, "reason": "answer_extraction_failed"}
            reference_parsed = math_verify.parse(f"${answer}$")
            prediction_parsed = math_verify.parse(f"${prediction}$")
            if not reference_parsed or not prediction_parsed:
                return {**result, "reason": "math_parse_failed"}
            correct = bool(math_verify.verify(reference_parsed, prediction_parsed))
        return {**result, "status": "correct" if correct else "incorrect", "reason": "original_math_verify"}
    except Exception as exc:
        return {**result, "reason": "evaluation_failed", "error_type": type(exc).__name__}
