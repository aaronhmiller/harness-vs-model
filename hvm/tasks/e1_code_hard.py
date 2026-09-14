"""E1 hard tier — tasks that a 3B-class model does NOT saturate.

Why this file exists: calibration showed the core 30 tasks pinned at the ceiling
for anything above ~1.5B (3B: 86.7%, with 22 of 30 tasks at 100%). A rung with
six partially-solved tasks has almost no room for a harness effect, so the whole
upper half of the ladder was dead weight.

These are harder in a specific way — each needs a small amount of *structure*
(a parser, a DP table, a heap, an eviction policy) rather than one clever line,
and each has an edge case that the obvious implementation gets wrong:
truncation toward zero, deterministic tie-breaks, escaped quotes, empty input,
capacity eviction order. That is deliberate: these are the tasks where an
execute-and-repair loop has the most to bite on.
"""
from __future__ import annotations

from typing import Any

E1_HARD_SPECS: list[dict[str, Any]] = [
    dict(
        id="h_eval_expr",
        fn="eval_expression",
        spec=(
            "Write `eval_expression(s)` that evaluates an arithmetic expression string "
            "containing non-negative integers, the binary operators + - * /, parentheses, "
            "and spaces. Normal precedence applies. Division is INTEGER division that "
            "truncates toward zero (so -7/2 is -3, not -4). The input is always valid. "
            "Do not use eval()."
        ),
        public=[
            'assert eval_expression("2+3*4") == 14',
            'assert eval_expression("(2+3)*4") == 20',
        ],
        hidden=[
            'assert eval_expression("7/2") == 3',
            'assert eval_expression("(0-7)/2") == -3',
            'assert eval_expression("1 + 2 * (3 - 1)") == 5',
            'assert eval_expression("100") == 100',
            'assert eval_expression("2*3+4*5") == 26',
            'assert eval_expression("((1+2)*(3+4))/5") == 4',
        ],
        solution='''
def eval_expression(s):
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[i:j]))
            i = j
        else:
            tokens.append(c)
            i += 1

    pos = 0

    def factor():
        nonlocal pos
        if tokens[pos] == "(":
            pos += 1
            v = expr()
            pos += 1
            return v
        v = tokens[pos]
        pos += 1
        return v

    def term():
        nonlocal pos
        v = factor()
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op = tokens[pos]
            pos += 1
            r = factor()
            if op == "*":
                v = v * r
            else:
                q = abs(v) // abs(r)
                v = q if (v < 0) == (r < 0) else -q
        return v

    def expr():
        nonlocal pos
        v = term()
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            op = tokens[pos]
            pos += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v

    return expr()
''',
    ),
    dict(
        id="h_lru",
        fn="LRUCache",
        spec=(
            "Write a class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` and "
            "`put(self, key, value)`. `get` returns the value or -1 if absent. Both `get` "
            "and `put` count as USING a key. When the cache is full, `put` evicts the least "
            "recently used key. A capacity of 0 or less raises ValueError."
        ),
        public=[
            '''
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
''',
            'assert LRUCache(1).get(9) == -1',
        ],
        hidden=[
            '''
c = LRUCache(2)
c.put(1, 1); c.put(2, 2); c.put(3, 3)
assert c.get(1) == -1 and c.get(2) == 2 and c.get(3) == 3
''',
            '''
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
c.get(1)            # 1 is now most recent
c.put(3, 3)         # evicts 2
assert c.get(2) == -1 and c.get(1) == 1
''',
            '''
c = LRUCache(2)
c.put(1, 1); c.put(1, 10)
assert c.get(1) == 10
''',
            '''
c = LRUCache(3)
for k in (1, 2, 3):
    c.put(k, k)
c.put(4, 4)
assert c.get(1) == -1 and c.get(4) == 4
''',
            '''
try:
    LRUCache(0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
class LRUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.cap = capacity
        self.d = {}

    def get(self, key):
        if key not in self.d:
            return -1
        v = self.d.pop(key)
        self.d[key] = v
        return v

    def put(self, key, value):
        if key in self.d:
            self.d.pop(key)
        elif len(self.d) >= self.cap:
            self.d.pop(next(iter(self.d)))
        self.d[key] = value
''',
    ),
    dict(
        id="h_topo",
        fn="topo_sort",
        spec=(
            "Write `topo_sort(deps)` where `deps` maps a task name to a list of task names "
            "that must come BEFORE it. Return a valid ordering of all tasks as a list. When "
            "several tasks are simultaneously available, choose the alphabetically smallest "
            "first, so the result is deterministic. Return None if there is a cycle. "
            "Tasks may appear only as prerequisites."
        ),
        public=[
            'assert topo_sort({"b": ["a"], "c": ["b"]}) == ["a", "b", "c"]',
            'assert topo_sort({"a": ["b"], "b": ["a"]}) is None',
        ],
        hidden=[
            'assert topo_sort({}) == []',
            'assert topo_sort({"z": [], "a": []}) == ["a", "z"]',
            'assert topo_sort({"d": ["b", "c"], "b": ["a"], "c": ["a"]}) == ["a", "b", "c", "d"]',
            'assert topo_sort({"x": ["y"]}) == ["y", "x"]',
            'assert topo_sort({"a": ["a"]}) is None',
        ],
        solution='''
import heapq

def topo_sort(deps):
    nodes = set(deps)
    for vs in deps.values():
        nodes |= set(vs)
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for n, pres in deps.items():
        for p in pres:
            adj[p].append(n)
            indeg[n] += 1
    heap = sorted(n for n in nodes if indeg[n] == 0)
    heapq.heapify(heap)
    out = []
    while heap:
        n = heapq.heappop(heap)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    return out if len(out) == len(nodes) else None
''',
    ),
    dict(
        id="h_wildcard",
        fn="wildcard_match",
        spec=(
            "Write `wildcard_match(s, p)` returning True if the whole string s matches the "
            "pattern p, where '*' matches any sequence of characters INCLUDING the empty "
            "one, and '?' matches exactly one character. All other characters match "
            "themselves. Do not use the re or fnmatch modules."
        ),
        public=[
            'assert wildcard_match("abc", "a*c") is True',
            'assert wildcard_match("abc", "a?c") is True',
        ],
        hidden=[
            'assert wildcard_match("", "*") is True',
            'assert wildcard_match("", "") is True',
            'assert wildcard_match("abc", "*") is True',
            'assert wildcard_match("abc", "?") is False',
            'assert wildcard_match("aa", "a") is False',
            'assert wildcard_match("adceb", "*a*b") is True',
            'assert wildcard_match("acdcb", "a*c?b") is False',
        ],
        solution='''
def wildcard_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif p[j - 1] == "?" or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
''',
    ),
    dict(
        id="h_simplify_path",
        fn="simplify_path",
        spec=(
            "Write `simplify_path(path)` that canonicalises an absolute Unix path. Collapse "
            "repeated slashes, drop '.', and resolve '..' by removing the previous "
            "component ('..' at the root is a no-op). The result must start with '/' and "
            "must not end with '/' unless it is exactly '/'."
        ),
        public=[
            'assert simplify_path("/a/./b//c/") == "/a/b/c"',
            'assert simplify_path("/../") == "/"',
        ],
        hidden=[
            'assert simplify_path("/") == "/"',
            'assert simplify_path("/a/../../b") == "/b"',
            'assert simplify_path("/home//foo/") == "/home/foo"',
            'assert simplify_path("/a/b/../c") == "/a/c"',
            'assert simplify_path("/...") == "/..."',
            'assert simplify_path("/a/..") == "/"',
        ],
        solution='''
def simplify_path(path):
    stack = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/" + "/".join(stack)
''',
    ),
    dict(
        id="h_multiply_strings",
        fn="multiply_strings",
        spec=(
            "Write `multiply_strings(a, b)` that multiplies two non-negative integers given "
            "as decimal strings and returns the product as a string. You must NOT convert "
            "the whole string with int() or use big-integer arithmetic directly — do the "
            "digit-by-digit multiplication. The result must have no leading zeros."
        ),
        public=[
            'assert multiply_strings("12", "12") == "144"',
            'assert multiply_strings("0", "999") == "0"',
        ],
        hidden=[
            'assert multiply_strings("2", "3") == "6"',
            'assert multiply_strings("123", "456") == "56088"',
            'assert multiply_strings("999", "999") == "998001"',
            'assert multiply_strings("1", "0") == "0"',
            'assert multiply_strings("100", "100") == "10000"',
            'assert multiply_strings("99999999", "99999999") == "9999999800000001"',
        ],
        solution='''
def multiply_strings(a, b):
    if a == "0" or b == "0":
        return "0"
    m, n = len(a), len(b)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = (ord(a[i]) - 48) * (ord(b[j]) - 48)
            p1, p2 = i + j, i + j + 1
            total = mul + res[p2]
            res[p2] = total % 10
            res[p1] += total // 10
    out = "".join(str(d) for d in res).lstrip("0")
    return out or "0"
''',
    ),
    dict(
        id="h_col_name",
        fn="spreadsheet_column_name",
        spec=(
            "Write `spreadsheet_column_name(n)` converting a 1-based column number to its "
            "spreadsheet letters: 1 -> 'A', 26 -> 'Z', 27 -> 'AA', 52 -> 'AZ', 53 -> 'BA'. "
            "Note this is bijective base-26 with NO zero digit. Raise ValueError for n < 1."
        ),
        public=[
            'assert spreadsheet_column_name(1) == "A"',
            'assert spreadsheet_column_name(27) == "AA"',
        ],
        hidden=[
            'assert spreadsheet_column_name(26) == "Z"',
            'assert spreadsheet_column_name(28) == "AB"',
            'assert spreadsheet_column_name(52) == "AZ"',
            'assert spreadsheet_column_name(53) == "BA"',
            'assert spreadsheet_column_name(702) == "ZZ"',
            'assert spreadsheet_column_name(703) == "AAA"',
            '''
try:
    spreadsheet_column_name(0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
def spreadsheet_column_name(n):
    if n < 1:
        raise ValueError("n must be >= 1")
    out = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        out.append(chr(65 + r))
    return "".join(reversed(out))
''',
    ),
    dict(
        id="h_csv_line",
        fn="parse_csv_line",
        spec=(
            "Write `parse_csv_line(line)` splitting ONE line of CSV into a list of field "
            "strings. A field may be wrapped in double quotes, in which case commas inside "
            "it are literal and a doubled quote (\"\") means one literal quote character. "
            "The surrounding quotes are not part of the value. An empty line yields ['']. "
            "Do not use the csv module."
        ),
        public=[
            'assert parse_csv_line("a,b,c") == ["a", "b", "c"]',
            'assert parse_csv_line(\'a,"b,c",d\') == ["a", "b,c", "d"]',
        ],
        hidden=[
            'assert parse_csv_line("") == [""]',
            'assert parse_csv_line("a,,b") == ["a", "", "b"]',
            'assert parse_csv_line(\'"say ""hi"""\') == [\'say "hi"\']',
            'assert parse_csv_line(\'"",x\') == ["", "x"]',
            'assert parse_csv_line("a,b,") == ["a", "b", ""]',
            'assert parse_csv_line(\'"a""b",c\') == [\'a"b\', "c"]',
        ],
        solution='''
def parse_csv_line(line):
    out, cur = [], []
    i, inq = 0, False
    while i < len(line):
        c = line[i]
        if inq:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 2
                    continue
                inq = False
                i += 1
                continue
            cur.append(c)
            i += 1
        else:
            if c == '"':
                inq = True
                i += 1
            elif c == ",":
                out.append("".join(cur))
                cur = []
                i += 1
            else:
                cur.append(c)
                i += 1
    out.append("".join(cur))
    return out
''',
    ),
    dict(
        id="h_next_perm",
        fn="next_permutation",
        spec=(
            "Write `next_permutation(nums)` returning a NEW list holding the next "
            "lexicographically greater permutation of the integers in `nums`. If no greater "
            "permutation exists, return the smallest one (fully ascending). The input must "
            "not be modified."
        ),
        public=[
            'assert next_permutation([1, 2, 3]) == [1, 3, 2]',
            'assert next_permutation([3, 2, 1]) == [1, 2, 3]',
        ],
        hidden=[
            'assert next_permutation([1, 1, 5]) == [1, 5, 1]',
            'assert next_permutation([1]) == [1]',
            'assert next_permutation([]) == []',
            'assert next_permutation([1, 3, 2]) == [2, 1, 3]',
            'assert next_permutation([2, 3, 1]) == [3, 1, 2]',
            '''
src = [1, 2, 3]
next_permutation(src)
assert src == [1, 2, 3], "input list was mutated"
''',
        ],
        solution='''
def next_permutation(nums):
    a = list(nums)
    n = len(a)
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a
''',
    ),
    dict(
        id="h_word_wrap",
        fn="word_wrap",
        spec=(
            "Write `word_wrap(text, width)` returning a list of lines. Split the text on "
            "whitespace and greedily pack as many words per line as fit within `width` "
            "characters, counting the single spaces between them. A word longer than "
            "`width` goes on a line of its own. Text with no words returns []. "
            "Raise ValueError if width <= 0."
        ),
        public=[
            'assert word_wrap("a b c", 3) == ["a b", "c"]',
            'assert word_wrap("", 5) == []',
        ],
        hidden=[
            'assert word_wrap("hello world", 5) == ["hello", "world"]',
            'assert word_wrap("hello world", 11) == ["hello world"]',
            'assert word_wrap("supercalifragilistic ok", 5) == ["supercalifragilistic", "ok"]',
            'assert word_wrap("   ", 4) == []',
            'assert word_wrap("aa bb cc dd", 5) == ["aa bb", "cc dd"]',
            '''
try:
    word_wrap("a", 0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
def word_wrap(text, width):
    if width <= 0:
        raise ValueError("width must be positive")
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines
''',
    ),
    dict(
        id="h_merge_k",
        fn="merge_k_sorted",
        spec=(
            "Write `merge_k_sorted(lists)` merging a list of already-sorted integer lists "
            "into one sorted list. Duplicates are preserved. Empty inner lists and an empty "
            "outer list are both allowed."
        ),
        public=[
            'assert merge_k_sorted([[1, 4], [2, 3]]) == [1, 2, 3, 4]',
            'assert merge_k_sorted([]) == []',
        ],
        hidden=[
            'assert merge_k_sorted([[], []]) == []',
            'assert merge_k_sorted([[1], [], [0]]) == [0, 1]',
            'assert merge_k_sorted([[1, 1], [1]]) == [1, 1, 1]',
            'assert merge_k_sorted([[-2, 0], [-3, 5]]) == [-3, -2, 0, 5]',
            'assert merge_k_sorted([[1, 2, 3]]) == [1, 2, 3]',
        ],
        solution='''
import heapq

def merge_k_sorted(lists):
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l[0], i, 0))
    out = []
    while heap:
        v, i, j = heapq.heappop(heap)
        out.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))
    return out
''',
    ),
    dict(
        id="h_business_days",
        fn="add_business_days",
        spec=(
            "Write `add_business_days(date_str, n)` where date_str is 'YYYY-MM-DD' and n is "
            "a non-negative integer. Return, in the same format, the date n business days "
            "later, where business days are Monday to Friday. n=0 returns the input date "
            "unchanged even if it falls on a weekend. Raise ValueError for negative n."
        ),
        public=[
            'assert add_business_days("2026-01-05", 1) == "2026-01-06"',
            'assert add_business_days("2026-01-05", 0) == "2026-01-05"',
        ],
        hidden=[
            'assert add_business_days("2026-01-09", 1) == "2026-01-12"',
            'assert add_business_days("2026-01-09", 5) == "2026-01-16"',
            'assert add_business_days("2026-01-10", 1) == "2026-01-12"',
            'assert add_business_days("2026-01-10", 0) == "2026-01-10"',
            'assert add_business_days("2026-01-30", 2) == "2026-02-03"',
            '''
try:
    add_business_days("2026-01-05", -1)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
from datetime import date, timedelta

def add_business_days(date_str, n):
    if n < 0:
        raise ValueError("n must be non-negative")
    y, m, d = (int(x) for x in date_str.split("-"))
    cur = date(y, m, d)
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur.isoformat()
''',
    ),
    dict(
        id="h_ranges",
        fn="group_consecutive_ranges",
        spec=(
            "Write `group_consecutive_ranges(nums)` that sorts and de-duplicates a list of "
            "integers, then summarises runs of consecutive values as strings: a run of one "
            "is just the number, a longer run is 'first-last'. Return the list of these "
            "strings in ascending order. An empty input returns []."
        ),
        public=[
            'assert group_consecutive_ranges([1, 2, 3, 5]) == ["1-3", "5"]',
            'assert group_consecutive_ranges([]) == []',
        ],
        hidden=[
            'assert group_consecutive_ranges([7]) == ["7"]',
            'assert group_consecutive_ranges([3, 1, 2]) == ["1-3"]',
            'assert group_consecutive_ranges([1, 1, 2]) == ["1-2"]',
            'assert group_consecutive_ranges([1, 3, 5]) == ["1", "3", "5"]',
            'assert group_consecutive_ranges([-2, -1, 4]) == ["-2--1", "4"]',
            'assert group_consecutive_ranges([1, 2, 4, 5, 6]) == ["1-2", "4-6"]',
        ],
        solution='''
def group_consecutive_ranges(nums):
    if not nums:
        return []
    s = sorted(set(nums))
    out = []
    start = prev = s[0]
    for x in s[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return out
''',
    ),
    dict(
        id="h_k_distinct",
        fn="longest_substring_k_distinct",
        spec=(
            "Write `longest_substring_k_distinct(s, k)` returning the LENGTH of the longest "
            "contiguous substring of s containing at most k distinct characters. "
            "Return 0 if k <= 0 or s is empty."
        ),
        public=[
            'assert longest_substring_k_distinct("eceba", 2) == 3',
            'assert longest_substring_k_distinct("", 3) == 0',
        ],
        hidden=[
            'assert longest_substring_k_distinct("aa", 1) == 2',
            'assert longest_substring_k_distinct("abc", 0) == 0',
            'assert longest_substring_k_distinct("abc", 5) == 3',
            'assert longest_substring_k_distinct("aabbcc", 2) == 4',
            'assert longest_substring_k_distinct("aabbcc", 3) == 6',
            'assert longest_substring_k_distinct("abaccc", 2) == 4',
        ],
        solution='''
def longest_substring_k_distinct(s, k):
    if k <= 0 or not s:
        return 0
    counts = {}
    left = 0
    best = 0
    for right, c in enumerate(s):
        counts[c] = counts.get(c, 0) + 1
        while len(counts) > k:
            counts[s[left]] -= 1
            if counts[s[left]] == 0:
                del counts[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
''',
    ),
    dict(
        id="h_version_cmp",
        fn="compare_versions",
        spec=(
            "Write `compare_versions(a, b)` comparing two dotted version strings. Return -1 "
            "if a < b, 1 if a > b, 0 if equal. Components are integers compared numerically, "
            "so '1.10' is greater than '1.9'. Missing trailing components count as 0, so "
            "'1.2' equals '1.2.0'. Leading zeros are allowed ('1.02' equals '1.2')."
        ),
        public=[
            'assert compare_versions("1.2", "1.10") == -1',
            'assert compare_versions("1.2", "1.2.0") == 0',
        ],
        hidden=[
            'assert compare_versions("1.0", "1") == 0',
            'assert compare_versions("2.1", "2.0.9") == 1',
            'assert compare_versions("1.02", "1.2") == 0',
            'assert compare_versions("0.1", "1.0") == -1',
            'assert compare_versions("1.0.0.1", "1") == 1',
            'assert compare_versions("13.37", "13.37") == 0',
        ],
        solution='''
def compare_versions(a, b):
    pa = [int(x) for x in a.split(".")]
    pb = [int(x) for x in b.split(".")]
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    for x, y in zip(pa, pb):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0
''',
    ),
]
