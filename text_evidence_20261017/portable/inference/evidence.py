"""Conservative, reference-free text evidence. Never executes candidate code."""
import ast
from fractions import Fraction
import re
import numpy as np
from common import parse_answer

FORMAT_NAMES = ['numeric_box_parser_conflict', 'number_prefixed_symbolic_box', 'terminal_math_operator']
ARITHMETIC_NAMES = ['log_distinct_checked_pairs', 'log_distinct_violations', 'violation_fraction', 'last_checked_pair_violates']
NAMES = FORMAT_NAMES + ARITHMETIC_NAMES
REL_TOL = Fraction(1, 1000)
ABS_TOL = Fraction(1, 1000000)
SPANS = re.compile(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<![\\$])\$(?!\$)([^\n$]*?)\$(?!\$)', re.S)

def _clean(expression):
    text = expression.strip()
    text = re.sub(r'(?<![\w)])\*\*(.*?)\*\*(?![\w(])', r'\1', text)
    text = text.replace('−', '-').replace('×', '*').replace('÷', '/')
    for command, replacement in [(r'\times', '*'), (r'\cdot', '*'), (r'\div', '/'),
                                  (r'\left', ''), (r'\right', ''), (r'\,', ' '),
                                  (r'\;', ' '), (r'\!', ''), (r'\$', ''), (r'\%', '%')]:
        text = text.replace(command, replacement)
    for _ in range(8):
        new = re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', text)
        new = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'(\1)', new)
        if new == text:
            break
        text = new
    text = text.replace('$', '').replace('&', '').strip().rstrip(';').strip()
    text = re.sub(r'(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])', lambda m: m.group().replace(',', ''), text)
    text = re.sub(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%', r'(\1/100)', text)
    text = text.replace('^', '**')
    if text.endswith('.'):
        text = text[:-1]
    return text.strip()

def numeric_value(expression):
    """Whitelist AST arithmetic, exact decimal/fraction values and bounded powers."""
    text = _clean(expression)
    if not text or len(text) > 240 or not re.fullmatch(r'[\d\s.+*/()\-]+', text):
        return None
    try:
        tree = ast.parse(text, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 80:
            return None
        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = Fraction(ast.get_source_segment(text, node))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = visit(node.operand)
                value = v if isinstance(node.op, ast.UAdd) else -v
            elif isinstance(node, ast.BinOp):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): value = a+b
                elif isinstance(node.op, ast.Sub): value = a-b
                elif isinstance(node.op, ast.Mult): value = a*b
                elif isinstance(node.op, ast.Div): value = a/b
                elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 8: value = a**int(b)
                else: raise ValueError('Unsupported operator')
            else:
                raise ValueError('Not a numeric expression')
            if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
                raise ValueError('Arithmetic limit')
            return value
        return visit(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError, TypeError):
        return None

def format_evidence(text):
    parsed = parse_answer(text)
    flags = [0., 0., 0.]
    info = {'official_parse': parsed, 'last_box': None, 'numeric_box_value': None}
    boxes = list(re.finditer(r'\\boxed\{([^{}]+)\}', text))
    if boxes:
        box = boxes[-1]
        token = box.group(1).strip()
        info['last_box'] = token
        if box.start() > text.rfind('####'):
            if re.fullmatch(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', token):
                try:
                    value = str(Fraction(token.replace(',', '')))
                    info['numeric_box_value'] = value
                    flags[0] = float(value != parsed)
                except (ValueError, ZeroDivisionError):
                    pass
            else:
                flags[1] = float(parsed is not None and bool(re.match(r'\s*[-+]?\d', token)) and bool(re.search(r'[A-Za-z]', token)))
    ending = re.sub(r'(?<![\w)])\*\*(.*?)\*\*(?![\w(])', r'\1', text, flags=re.S).rstrip()
    ending = re.sub(r'(?:\$\$|\$|\\\]|\\\)|```)+\s*$', '', ending).rstrip()
    flags[2] = float(bool(re.search(r'[=+*/]\s*$', ending)))
    return flags, info

def arithmetic_evidence(text):
    sources = []
    for match in SPANS.finditer(text):
        # Units written just outside a math delimiter can change numeric scale ($5=500$ cents).
        if re.match(r'[ \t]*[A-Za-z0-9]',text[match.end():]):
            continue
        body = next(v for v in match.groups() if v is not None)
        sources.append((match.start(), body))
    offset = 0
    for line in text.splitlines(keepends=True):
        # A full plain equation or a label followed by a full equation is supported.
        candidate = re.sub(r'^\s*[-*]\s+', '', line.strip())
        if ':' in candidate and '=' not in candidate.split(':', 1)[0]:
            candidate = candidate.split(':', 1)[1]
        sources.append((offset, candidate))
        offset += len(line)
    seen, pairs = set(), []
    unknown_pairs = 0
    for offset, body in sorted(sources, key=lambda r:r[0]):
        body = body.replace(r'\\', '\n')
        for line in body.splitlines():
            line = re.sub(r'(?<![\w)])\*\*(.*?)\*\*(?![\w(])', r'\1', line)
            if '=' not in line or re.search(r'<=|>=|!=|≈|\\approx|\\sim', line):
                continue
            pieces = line.split('=')
            values = [numeric_value(s) for s in pieces]
            for a, b, av, bv in zip(pieces, pieces[1:], values, values[1:]):
                # Percentage notation is convention-dependent (ratio*100=50%); abstain.
                if av is None or bv is None or '%' in a or '%' in b:
                    unknown_pairs += 1
                    continue
                key = (_clean(a), _clean(b))
                if key in seen:
                    continue
                seen.add(key)
                tolerance = ABS_TOL + REL_TOL * max(abs(av), abs(bv))
                violation = abs(av-bv) > tolerance
                pairs.append({'left': a.strip(), 'right': b.strip(), 'normalized_left': key[0], 'normalized_right': key[1],
                    'left_value': str(av), 'right_value': str(bv), 'violates': bool(violation), 'source_offset': offset})
                if len(pairs) >= 256:
                    break
            if len(pairs) >= 256:
                break
        if len(pairs) >= 256:
            break
    n, errors = len(pairs), sum(p['violates'] for p in pairs)
    features = [float(np.log1p(n)), float(np.log1p(errors)), errors/n if n else 0., float(pairs[-1]['violates']) if n else 0.]
    return features, {'checked_pairs': n, 'violations': errors, 'unknown_pair_observations': unknown_pairs, 'pairs': pairs}

def candidate_evidence(text):
    f, fi = format_evidence(text)
    a, ai = arithmetic_evidence(text)
    return np.asarray(f+a, np.float64), {'format': fi, 'arithmetic': ai}

def extract(bank, detailed=False):
    result = np.zeros((len(bank), 4, len(NAMES)), np.float64)
    details = []
    for q, row in enumerate(bank):
        for i, candidate in enumerate(row['candidates']):
            result[q,i], info = candidate_evidence(candidate['text'])
            if detailed:
                details.append({'row': q, 'candidate': i, **info})
    assert np.isfinite(result).all()
    return (result, details) if detailed else result
