"""E1 -- the verifiable eval: write a function, graded by hidden tests.

Test split discipline (this is load-bearing):
  * `public` tests are shown to H2's repair loop when a candidate fails.
  * `hidden` tests decide the score and are NEVER shown to any harness.
Both sets probe the SAME spec. Hidden tests add edge cases, not new
requirements -- otherwise the split would penalise H2 for obeying its feedback.

Difficulty is deliberately aimed at a ~30-60% single-shot pass rate for a
1.5B-class model: each task has at least one edge case (empty input, negatives,
tie-breaking, leading zeros, acronyms) that small models routinely miss.
"""
from __future__ import annotations

from typing import Any

E1_SPECS: list[dict[str, Any]] = [
    dict(
        id="rle",
        fn="run_length_encode",
        spec=(
            "Write `run_length_encode(s)` that takes a string and returns a list of "
            "(character, count) tuples for each run of consecutive identical characters, "
            "in order. An empty string returns an empty list. Matching is case-sensitive."
        ),
        public=[
            'assert run_length_encode("aaabb") == [("a", 3), ("b", 2)]',
            'assert run_length_encode("") == []',
        ],
        hidden=[
            'assert run_length_encode("a") == [("a", 1)]',
            'assert run_length_encode("abab") == [("a",1),("b",1),("a",1),("b",1)]',
            'assert run_length_encode("aAaa") == [("a",1),("A",1),("a",2)]',
            'assert run_length_encode("zzzzzzz") == [("z", 7)]',
            'assert run_length_encode("  x") == [(" ", 2), ("x", 1)]',
        ],
        solution='''
def run_length_encode(s):
    out = []
    for ch in s:
        if out and out[-1][0] == ch:
            out[-1] = (ch, out[-1][1] + 1)
        else:
            out.append((ch, 1))
    return out
''',
    ),
    dict(
        id="brackets",
        fn="balanced_brackets",
        spec=(
            "Write `balanced_brackets(s)` returning True if every (), [] and {} in the "
            "string is correctly matched and nested. Characters that are not brackets are "
            "ignored. The empty string is balanced."
        ),
        public=[
            'assert balanced_brackets("([]{})") is True',
            'assert balanced_brackets("(]") is False',
        ],
        hidden=[
            'assert balanced_brackets("") is True',
            'assert balanced_brackets("a(b)c[d]{e}") is True',
            'assert balanced_brackets("(") is False',
            'assert balanced_brackets(")(") is False',
            'assert balanced_brackets("{[()]}") is True',
            'assert balanced_brackets("{[(])}") is False',
        ],
        solution='''
def balanced_brackets(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''',
    ),
    dict(
        id="second_largest",
        fn="second_largest",
        spec=(
            "Write `second_largest(nums)` returning the second largest DISTINCT value in a "
            "list of integers, or None if there are fewer than two distinct values."
        ),
        public=[
            'assert second_largest([1, 3, 2]) == 2',
            'assert second_largest([5, 5]) is None',
        ],
        hidden=[
            'assert second_largest([]) is None',
            'assert second_largest([7]) is None',
            'assert second_largest([4, 4, 3, 3]) == 3',
            'assert second_largest([-1, -2, -3]) == -2',
            'assert second_largest([10, 9, 10, 9, 8]) == 9',
        ],
        solution='''
def second_largest(nums):
    d = sorted(set(nums), reverse=True)
    return d[1] if len(d) >= 2 else None
''',
    ),
    dict(
        id="merge_intervals",
        fn="merge_intervals",
        spec=(
            "Write `merge_intervals(intervals)` taking a list of (start, end) tuples and "
            "returning a new list of non-overlapping (start, end) tuples sorted by start. "
            "Intervals that merely touch (end == next start) must be merged. "
            "An empty input returns an empty list."
        ),
        public=[
            'assert merge_intervals([(1,3),(2,6),(8,10)]) == [(1,6),(8,10)]',
            'assert merge_intervals([]) == []',
        ],
        hidden=[
            'assert merge_intervals([(1,4),(4,5)]) == [(1,5)]',
            'assert merge_intervals([(5,6),(1,2)]) == [(1,2),(5,6)]',
            'assert merge_intervals([(1,10),(2,3)]) == [(1,10)]',
            'assert merge_intervals([(1,2)]) == [(1,2)]',
            'assert merge_intervals([(1,4),(0,4)]) == [(0,4)]',
        ],
        solution='''
def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals, key=lambda x: (x[0], x[1]))
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]
''',
    ),
    dict(
        id="word_freq",
        fn="word_frequencies",
        spec=(
            "Write `word_frequencies(text)` returning a dict mapping each word to its count. "
            "A word is a maximal run of ASCII letters or digits; everything else is a "
            "separator. Words are lowercased before counting."
        ),
        public=[
            'assert word_frequencies("Hi hi there") == {"hi": 2, "there": 1}',
            'assert word_frequencies("") == {}',
        ],
        hidden=[
            'assert word_frequencies("a-b a") == {"a": 2, "b": 1}',
            'assert word_frequencies("Dog, dog. DOG!") == {"dog": 3}',
            'assert word_frequencies("x1 x1 y") == {"x1": 2, "y": 1}',
            'assert word_frequencies("...") == {}',
            'assert word_frequencies("One") == {"one": 1}',
        ],
        solution='''
import re

def word_frequencies(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    d = {}
    for w in words:
        d[w] = d.get(w, 0) + 1
    return d
''',
    ),
    dict(
        id="is_rotation",
        fn="is_rotation",
        spec=(
            "Write `is_rotation(a, b)` returning True if string b is a rotation of string a "
            "(rotating 'abcd' can give 'cdab'). Strings of different lengths are never "
            "rotations. Two empty strings ARE rotations of each other."
        ),
        public=[
            'assert is_rotation("abcd", "cdab") is True',
            'assert is_rotation("abcd", "abdc") is False',
        ],
        hidden=[
            'assert is_rotation("", "") is True',
            'assert is_rotation("a", "a") is True',
            'assert is_rotation("abc", "abcabc") is False',
            'assert is_rotation("aab", "aba") is True',
            'assert is_rotation("abc", "") is False',
        ],
        solution='''
def is_rotation(a, b):
    return len(a) == len(b) and b in (a + a)
''',
    ),
    dict(
        id="chunk",
        fn="chunk",
        spec=(
            "Write `chunk(lst, n)` splitting a list into consecutive sublists of length n. "
            "The final chunk may be shorter. If n <= 0, raise ValueError. "
            "An empty list returns an empty list."
        ),
        public=[
            'assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]',
            'assert chunk([], 3) == []',
        ],
        hidden=[
            'assert chunk([1,2,3], 5) == [[1,2,3]]',
            'assert chunk([1,2,3,4], 4) == [[1,2,3,4]]',
            'assert chunk([1,2,3], 1) == [[1],[2],[3]]',
            '''
try:
    chunk([1,2,3], 0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
            '''
try:
    chunk([1,2,3], -2)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
def chunk(lst, n):
    if n <= 0:
        raise ValueError("n must be positive")
    return [lst[i:i + n] for i in range(0, len(lst), n)]
''',
    ),
    dict(
        id="flatten",
        fn="flatten",
        spec=(
            "Write `flatten(nested)` that fully flattens arbitrarily nested lists and tuples "
            "into a single flat list, preserving order. Strings are NOT flattened -- they are "
            "treated as atomic values."
        ),
        public=[
            'assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]',
            'assert flatten([]) == []',
        ],
        hidden=[
            'assert flatten([[[[1]]]]) == [1]',
            'assert flatten(["ab", ["cd"]]) == ["ab", "cd"]',
            'assert flatten([1, (2, 3), [4]]) == [1, 2, 3, 4]',
            'assert flatten([[], [], [1]]) == [1]',
            'assert flatten([None, [False]]) == [None, False]',
        ],
        solution='''
def flatten(nested):
    out = []
    for x in nested:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''',
    ),
    dict(
        id="roman_to_int",
        fn="roman_to_int",
        spec=(
            "Write `roman_to_int(s)` converting a valid Roman numeral string (I, V, X, L, C, "
            "D, M) to an integer, handling subtractive pairs such as IV, IX, XL, XC, CD, CM."
        ),
        public=[
            'assert roman_to_int("IX") == 9',
            'assert roman_to_int("XIII") == 13',
        ],
        hidden=[
            'assert roman_to_int("I") == 1',
            'assert roman_to_int("MCMXCIV") == 1994',
            'assert roman_to_int("MMMCMXCIX") == 3999',
            'assert roman_to_int("XL") == 40',
            'assert roman_to_int("CDXLIV") == 444',
        ],
        solution='''
def roman_to_int(s):
    vals = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total
''',
    ),
    dict(
        id="int_to_roman",
        fn="int_to_roman",
        spec=(
            "Write `int_to_roman(n)` converting an integer in the range 1..3999 to its "
            "standard Roman numeral string, using subtractive forms (4 -> IV, 9 -> IX, "
            "40 -> XL, 90 -> XC, 400 -> CD, 900 -> CM)."
        ),
        public=[
            'assert int_to_roman(9) == "IX"',
            'assert int_to_roman(13) == "XIII"',
        ],
        hidden=[
            'assert int_to_roman(1) == "I"',
            'assert int_to_roman(1994) == "MCMXCIV"',
            'assert int_to_roman(3999) == "MMMCMXCIX"',
            'assert int_to_roman(40) == "XL"',
            'assert int_to_roman(444) == "CDXLIV"',
        ],
        solution='''
def int_to_roman(n):
    table = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
             (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    out = []
    for v, sym in table:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)
''',
    ),
    dict(
        id="lcp",
        fn="longest_common_prefix",
        spec=(
            "Write `longest_common_prefix(strs)` returning the longest string that is a "
            "prefix of every string in the list. Return \"\" if the list is empty or there is "
            "no common prefix."
        ),
        public=[
            'assert longest_common_prefix(["flower","flow","flight"]) == "fl"',
            'assert longest_common_prefix([]) == ""',
        ],
        hidden=[
            'assert longest_common_prefix(["dog","racecar"]) == ""',
            'assert longest_common_prefix(["abc"]) == "abc"',
            'assert longest_common_prefix(["", "abc"]) == ""',
            'assert longest_common_prefix(["aa","aa"]) == "aa"',
            'assert longest_common_prefix(["ab","abc","abcd"]) == "ab"',
        ],
        solution='''
def longest_common_prefix(strs):
    if not strs:
        return ""
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p):
            p = p[:-1]
            if not p:
                return ""
    return p
''',
    ),
    dict(
        id="move_zeros",
        fn="move_zeros",
        spec=(
            "Write `move_zeros(nums)` returning a NEW list where all zeros are moved to the "
            "end while the relative order of the non-zero integers is preserved. "
            "The input list must not be modified."
        ),
        public=[
            'assert move_zeros([0,1,0,3,12]) == [1,3,12,0,0]',
            'assert move_zeros([]) == []',
        ],
        hidden=[
            'assert move_zeros([0,0,0]) == [0,0,0]',
            'assert move_zeros([1,2,3]) == [1,2,3]',
            'assert move_zeros([-1,0,-2]) == [-1,-2,0]',
            '''
src = [0, 1]
move_zeros(src)
assert src == [0, 1], "input list was mutated"
''',
            'assert move_zeros([0]) == [0]',
        ],
        solution='''
def move_zeros(nums):
    nz = [x for x in nums if x != 0]
    return nz + [0] * (len(nums) - len(nz))
''',
    ),
    dict(
        id="spiral",
        fn="spiral_order",
        spec=(
            "Write `spiral_order(matrix)` returning the elements of a rectangular 2D list in "
            "clockwise spiral order starting at the top-left. An empty matrix (or a matrix "
            "with empty rows) returns an empty list."
        ),
        public=[
            'assert spiral_order([[1,2],[3,4]]) == [1,2,4,3]',
            'assert spiral_order([]) == []',
        ],
        hidden=[
            'assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]',
            'assert spiral_order([[1,2,3]]) == [1,2,3]',
            'assert spiral_order([[1],[2],[3]]) == [1,2,3]',
            'assert spiral_order([[]]) == []',
            'assert spiral_order([[1,2,3,4],[5,6,7,8]]) == [1,2,3,4,8,7,6,5]',
        ],
        solution='''
def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    out = []
    top, bot = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bot and left <= right:
        for c in range(left, right + 1):
            out.append(matrix[top][c])
        top += 1
        for r in range(top, bot + 1):
            out.append(matrix[r][right])
        right -= 1
        if top <= bot:
            for c in range(right, left - 1, -1):
                out.append(matrix[bot][c])
            bot -= 1
        if left <= right:
            for r in range(bot, top - 1, -1):
                out.append(matrix[r][left])
            left += 1
    return out
''',
    ),
    dict(
        id="top_k",
        fn="top_k_frequent",
        spec=(
            "Write `top_k_frequent(nums, k)` returning the k most frequent integers, ordered "
            "by descending frequency. Ties in frequency are broken by SMALLER value first. "
            "If k exceeds the number of distinct values, return all of them."
        ),
        public=[
            'assert top_k_frequent([1,1,2,2,3], 2) == [1,2]',
            'assert top_k_frequent([], 3) == []',
        ],
        hidden=[
            'assert top_k_frequent([3,3,1,1,2], 2) == [1,3]',
            'assert top_k_frequent([5,4,3], 2) == [3,4]',
            'assert top_k_frequent([1,2,3], 10) == [1,2,3]',
            'assert top_k_frequent([7,7,7], 1) == [7]',
            'assert top_k_frequent([2,2,1,1,3,3], 3) == [1,2,3]',
        ],
        solution='''
from collections import Counter

def top_k_frequent(nums, k):
    c = Counter(nums)
    ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return [v for v, _ in ordered[:k]]
''',
    ),
    dict(
        id="palindrome",
        fn="is_palindrome_alnum",
        spec=(
            "Write `is_palindrome_alnum(s)` returning True if the string reads the same "
            "forwards and backwards once non-alphanumeric characters are removed and case is "
            "ignored. The empty string is a palindrome."
        ),
        public=[
            'assert is_palindrome_alnum("A man, a plan, a canal: Panama") is True',
            'assert is_palindrome_alnum("hello") is False',
        ],
        hidden=[
            'assert is_palindrome_alnum("") is True',
            'assert is_palindrome_alnum(".,!") is True',
            'assert is_palindrome_alnum("0P") is False',
            'assert is_palindrome_alnum("aba") is True',
            'assert is_palindrome_alnum("12321") is True',
        ],
        solution='''
def is_palindrome_alnum(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
''',
    ),
    dict(
        id="anagrams",
        fn="group_anagrams",
        spec=(
            "Write `group_anagrams(words)` grouping words that are anagrams of each other. "
            "Return a list of groups where each group is sorted alphabetically, and the list "
            "of groups is itself sorted. An empty input returns an empty list."
        ),
        public=[
            'assert group_anagrams(["eat","tea","tan"]) == [["eat","tea"],["tan"]]',
            'assert group_anagrams([]) == []',
        ],
        hidden=[
            'assert group_anagrams(["abc","cba","bca"]) == [["abc","bca","cba"]]',
            'assert group_anagrams(["a"]) == [["a"]]',
            'assert group_anagrams(["ab","ba","abc"]) == [["ab","ba"],["abc"]]',
            'assert group_anagrams(["",""]) == [["",""]]',
            'assert group_anagrams(["xy","yx","zz"]) == [["xy","yx"],["zz"]]',
        ],
        solution='''
def group_anagrams(words):
    d = {}
    for w in words:
        d.setdefault("".join(sorted(w)), []).append(w)
    return sorted(sorted(g) for g in d.values())
''',
    ),
    dict(
        id="bisect",
        fn="binary_search_insert",
        spec=(
            "Write `binary_search_insert(arr, target)` returning the index at which target "
            "should be inserted into the already-sorted list `arr` to keep it sorted. If "
            "target already occurs, return the index of its FIRST occurrence. "
            "Do not use the bisect module."
        ),
        public=[
            'assert binary_search_insert([1,3,5], 4) == 2',
            'assert binary_search_insert([], 1) == 0',
        ],
        hidden=[
            'assert binary_search_insert([1,2,2,2,3], 2) == 1',
            'assert binary_search_insert([1,2,3], 0) == 0',
            'assert binary_search_insert([1,2,3], 9) == 3',
            'assert binary_search_insert([5], 5) == 0',
            'assert binary_search_insert([-3,-1,0], -2) == 1',
        ],
        solution='''
def binary_search_insert(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
    ),
    dict(
        id="transpose",
        fn="matrix_transpose",
        spec=(
            "Write `matrix_transpose(m)` returning the transpose of a rectangular 2D list as "
            "a list of lists. An empty matrix, or one whose rows are empty, returns []."
        ),
        public=[
            'assert matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]]',
            'assert matrix_transpose([]) == []',
        ],
        hidden=[
            'assert matrix_transpose([[1,2,3]]) == [[1],[2],[3]]',
            'assert matrix_transpose([[1],[2]]) == [[1,2]]',
            'assert matrix_transpose([[]]) == []',
            'assert matrix_transpose([[1,2],[3,4],[5,6]]) == [[1,3,5],[2,4,6]]',
            'assert matrix_transpose([[0]]) == [[0]]',
        ],
        solution='''
def matrix_transpose(m):
    if not m or not m[0]:
        return []
    return [list(row) for row in zip(*m)]
''',
    ),
    dict(
        id="islands",
        fn="count_islands",
        spec=(
            "Write `count_islands(grid)` counting connected groups of 1s in a 2D grid of 0s "
            "and 1s. Cells connect ORTHOGONALLY only (up/down/left/right), not diagonally. "
            "An empty grid returns 0."
        ),
        public=[
            'assert count_islands([[1,0],[0,1]]) == 2',
            'assert count_islands([]) == 0',
        ],
        hidden=[
            'assert count_islands([[1,1],[1,1]]) == 1',
            'assert count_islands([[0,0],[0,0]]) == 0',
            'assert count_islands([[1,0,1],[0,0,0],[1,0,1]]) == 4',
            'assert count_islands([[1,1,0],[0,1,0],[0,0,1]]) == 2',
            'assert count_islands([[1]]) == 1',
        ],
        solution='''
def count_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in seen:
                count += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            if grid[ny][nx] == 1 and (ny, nx) not in seen:
                                seen.add((ny, nx))
                                stack.append((ny, nx))
    return count
''',
    ),
    dict(
        id="fib",
        fn="fib",
        spec=(
            "Write `fib(n)` returning the n-th Fibonacci number where fib(0) == 0 and "
            "fib(1) == 1. It must handle n up to 200 quickly. "
            "If n is negative, raise ValueError."
        ),
        public=[
            'assert fib(10) == 55',
            'assert fib(0) == 0',
        ],
        hidden=[
            'assert fib(1) == 1',
            'assert fib(2) == 1',
            'assert fib(50) == 12586269025',
            'assert fib(200) == 280571172992510140037611932413038677189525',
            '''
try:
    fib(-1)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
def fib(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    ),
    dict(
        id="querystring",
        fn="parse_query_string",
        spec=(
            "Write `parse_query_string(qs)` parsing a URL query string into a dict. "
            "'a=1&b=2' -> {'a': '1', 'b': '2'}. A key that appears more than once maps to a "
            "LIST of its values in order: 'a=1&a=2' -> {'a': ['1','2']}. A key with no '=' "
            "maps to ''. An empty string returns {}. Do not use urllib."
        ),
        public=[
            'assert parse_query_string("a=1&b=2") == {"a": "1", "b": "2"}',
            'assert parse_query_string("") == {}',
        ],
        hidden=[
            'assert parse_query_string("a=1&a=2") == {"a": ["1","2"]}',
            'assert parse_query_string("a=1&a=2&a=3") == {"a": ["1","2","3"]}',
            'assert parse_query_string("flag") == {"flag": ""}',
            'assert parse_query_string("a=") == {"a": ""}',
            'assert parse_query_string("a=1=2") == {"a": "1=2"}',
        ],
        solution='''
def parse_query_string(qs):
    out = {}
    if not qs:
        return out
    for part in qs.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        if k in out:
            if isinstance(out[k], list):
                out[k].append(v)
            else:
                out[k] = [out[k], v]
        else:
            out[k] = v
    return out
''',
    ),
    dict(
        id="camel_snake",
        fn="camel_to_snake",
        spec=(
            "Write `camel_to_snake(s)` converting camelCase/PascalCase to snake_case. "
            "Consecutive capitals forming an acronym stay together: "
            "'HTTPServer' -> 'http_server', 'parseJSON' -> 'parse_json', "
            "'getHTTPResponseCode' -> 'get_http_response_code'."
        ),
        public=[
            'assert camel_to_snake("camelCase") == "camel_case"',
            'assert camel_to_snake("HTTPServer") == "http_server"',
        ],
        hidden=[
            'assert camel_to_snake("parseJSON") == "parse_json"',
            'assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"',
            'assert camel_to_snake("simpleXML") == "simple_xml"',
            'assert camel_to_snake("lower") == "lower"',
            'assert camel_to_snake("PascalCase") == "pascal_case"',
        ],
        solution='''
import re

def camel_to_snake(s):
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s)
    return s.lower()
''',
    ),
    dict(
        id="dedupe",
        fn="dedupe_preserve_order",
        spec=(
            "Write `dedupe_preserve_order(lst)` returning a new list with duplicates removed, "
            "keeping the FIRST occurrence of each value and the original relative order."
        ),
        public=[
            'assert dedupe_preserve_order([3,1,3,2,1]) == [3,1,2]',
            'assert dedupe_preserve_order([]) == []',
        ],
        hidden=[
            'assert dedupe_preserve_order([1,1,1]) == [1]',
            'assert dedupe_preserve_order(["b","a","b"]) == ["b","a"]',
            'assert dedupe_preserve_order([1,2,3]) == [1,2,3]',
            'assert dedupe_preserve_order([None, None, 0]) == [None, 0]',
            'assert dedupe_preserve_order(["x"]) == ["x"]',
        ],
        solution='''
def dedupe_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
    ),
    dict(
        id="kadane",
        fn="max_subarray_sum",
        spec=(
            "Write `max_subarray_sum(nums)` returning the largest sum of any CONTIGUOUS "
            "non-empty subarray. If every number is negative, return the largest single "
            "element. For an empty list, return 0."
        ),
        public=[
            'assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6',
            'assert max_subarray_sum([]) == 0',
        ],
        hidden=[
            'assert max_subarray_sum([-3,-1,-7]) == -1',
            'assert max_subarray_sum([5]) == 5',
            'assert max_subarray_sum([-5]) == -5',
            'assert max_subarray_sum([1,2,3]) == 6',
            'assert max_subarray_sum([0,-1,0]) == 0',
        ],
        solution='''
def max_subarray_sum(nums):
    if not nums:
        return 0
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
''',
    ),
    dict(
        id="ipv4",
        fn="validate_ipv4",
        spec=(
            "Write `validate_ipv4(s)` returning True only for a valid dotted-quad IPv4 "
            "address: exactly four parts, each all decimal digits, each 0-255, and with NO "
            "leading zeros (so '01' is invalid but '0' is valid)."
        ),
        public=[
            'assert validate_ipv4("192.168.0.1") is True',
            'assert validate_ipv4("256.1.1.1") is False',
        ],
        hidden=[
            'assert validate_ipv4("0.0.0.0") is True',
            'assert validate_ipv4("01.1.1.1") is False',
            'assert validate_ipv4("1.1.1") is False',
            'assert validate_ipv4("1.1.1.1.1") is False',
            'assert validate_ipv4("1.1.1.a") is False',
            'assert validate_ipv4("255.255.255.255") is True',
            'assert validate_ipv4("1.1.1.-1") is False',
        ],
        solution='''
def validate_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or not all(c in "0123456789" for c in p):
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True
''',
    ),
    dict(
        id="min_coins",
        fn="min_coins",
        spec=(
            "Write `min_coins(coins, amount)` returning the fewest coins summing exactly to "
            "amount, or -1 if impossible. Each denomination may be used unlimited times. "
            "An amount of 0 needs 0 coins."
        ),
        public=[
            'assert min_coins([1,5,10], 12) == 3',
            'assert min_coins([2], 3) == -1',
        ],
        hidden=[
            'assert min_coins([1,2,5], 11) == 3',
            'assert min_coins([], 0) == 0',
            'assert min_coins([5], 0) == 0',
            'assert min_coins([1,3,4], 6) == 2',
            'assert min_coins([7], 14) == 2',
            'assert min_coins([2,5], 3) == -1',
        ],
        solution='''
def min_coins(coins, amount):
    if amount == 0:
        return 0
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]
''',
    ),
    dict(
        id="rotate",
        fn="rotate_list",
        spec=(
            "Write `rotate_list(lst, k)` returning a NEW list rotated RIGHT by k positions. "
            "k may be larger than the list length, and a negative k rotates LEFT. "
            "An empty list returns an empty list."
        ),
        public=[
            'assert rotate_list([1,2,3,4,5], 2) == [4,5,1,2,3]',
            'assert rotate_list([], 3) == []',
        ],
        hidden=[
            'assert rotate_list([1,2,3], 0) == [1,2,3]',
            'assert rotate_list([1,2,3], 3) == [1,2,3]',
            'assert rotate_list([1,2,3], 4) == [3,1,2]',
            'assert rotate_list([1,2,3], -1) == [2,3,1]',
            'assert rotate_list([1,2,3], -4) == [2,3,1]',
        ],
        solution='''
def rotate_list(lst, k):
    if not lst:
        return []
    k = k % len(lst)
    if k == 0:
        return list(lst)
    return lst[-k:] + lst[:-k]
''',
    ),
    dict(
        id="interleave",
        fn="interleave",
        spec=(
            "Write `interleave(a, b)` returning a list that alternates elements of a and b, "
            "starting with a. When one list runs out, append the remainder of the other."
        ),
        public=[
            'assert interleave([1,3], [2,4]) == [1,2,3,4]',
            'assert interleave([], []) == []',
        ],
        hidden=[
            'assert interleave([1,2,3], [9]) == [1,9,2,3]',
            'assert interleave([1], [7,8,9]) == [1,7,8,9]',
            'assert interleave([], [1,2]) == [1,2]',
            'assert interleave([1,2], []) == [1,2]',
            'assert interleave(["a"], ["b"]) == ["a","b"]',
        ],
        solution='''
def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out
''',
    ),
    dict(
        id="digital_root",
        fn="digital_root",
        spec=(
            "Write `digital_root(n)` repeatedly summing the decimal digits of a non-negative "
            "integer until a single digit remains, and returning it. digital_root(0) == 0. "
            "Raise ValueError for negative input."
        ),
        public=[
            'assert digital_root(38) == 2',
            'assert digital_root(0) == 0',
        ],
        hidden=[
            'assert digital_root(9) == 9',
            'assert digital_root(10) == 1',
            'assert digital_root(99999) == 9',
            'assert digital_root(12345) == 6',
            '''
try:
    digital_root(-5)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
''',
        ],
        solution='''
def digital_root(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
''',
    ),
    dict(
        id="duplicates",
        fn="find_duplicates",
        spec=(
            "Write `find_duplicates(nums)` returning a SORTED list of the distinct values "
            "that appear more than once in the input list of integers. "
            "Return [] when there are no duplicates."
        ),
        public=[
            'assert find_duplicates([1,2,2,3,3,3]) == [2,3]',
            'assert find_duplicates([1,2,3]) == []',
        ],
        hidden=[
            'assert find_duplicates([]) == []',
            'assert find_duplicates([5,5,5,5]) == [5]',
            'assert find_duplicates([3,1,3,1]) == [1,3]',
            'assert find_duplicates([-1,-1,0]) == [-1]',
            'assert find_duplicates([0,0]) == [0]',
        ],
        solution='''
from collections import Counter

def find_duplicates(nums):
    return sorted(v for v, c in Counter(nums).items() if c > 1)
''',
    ),
]
