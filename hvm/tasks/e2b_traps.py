"""E2b -- scenario Y: self-verification-gated tasks.

Every item has an attractive wrong answer that a fast, confident reasoner
reaches. There is no external oracle, so H2's "check" step is the model grading
itself with the same intuition that produced the error. Prediction:
delta_harness ~= 0 or NEGATIVE, because self-consistency across samples
*stabilises* the trap answer rather than catching it.

Two design choices matter:

  * Numbers are varied from the canonical folklore versions (the bat-and-ball
    problem is $2.60/$2.00 here, not $1.10/$1.00). A model that has merely
    memorised "5 cents" scores zero. This separates reasoning from recall --
    without it, E2b would quietly become a second knowledge eval.

  * `trap` records the attractive wrong answer explicitly, so we can measure
    TRAP CAPTURE RATE, not just accuracy. If H2's capture rate exceeds H1's,
    that is direct evidence the loop amplified a counterfeit signal.
"""
from __future__ import annotations

from typing import Any

E2B_SPECS: list[dict[str, Any]] = [
    dict(id="t01", numeric=True,
         q="A racket and a ball cost $2.60 in total. The racket costs $2.00 more than the ball. How much does the ball cost, in dollars?",
         a=["0.30"], trap=["0.60"]),
    dict(id="t02", numeric=True,
         q="If 9 machines take 9 minutes to make 9 widgets, how many minutes would 90 machines take to make 90 widgets?",
         a=["9"], trap=["90"]),
    dict(id="t03", numeric=True,
         q="A patch of lily pads doubles in size every day. It covers the entire lake on day 30. On which day did it cover half the lake?",
         a=["29"], trap=["15"]),
    dict(id="t04", numeric=True,
         q="A farmer has 17 sheep. All but 9 die. How many sheep are left?",
         a=["9"], trap=["8"]),
    dict(id="t05", numeric=True,
         q="How many months of the year have at least 28 days?",
         a=["12"], trap=["1"]),
    dict(id="t06", numeric=True,
         q="A car drives 60 miles at 30 mph, then 60 miles at 60 mph. What is its average speed for the whole trip, in mph?",
         a=["40"], trap=["45"]),
    dict(id="t07", numeric=False,
         q="You are running a race and you overtake the runner in second place. What position are you in now? Answer with an ordinal such as 'first' or 'second'.",
         a=["second", "2nd", "2"], trap=["first", "1st", "1"]),
    dict(id="t08", numeric=True,
         q="A clock takes 6 seconds to strike 4 o'clock (four strikes). How many seconds does it take to strike 8 o'clock?",
         a=["14"], trap=["12"]),
    dict(id="t09", numeric=True,
         q="Divide 30 by one half, then add 10. What is the result?",
         a=["70"], trap=["25"]),
    dict(id="t10", numeric=False,
         q="Mary's father has five daughters: Nana, Nene, Nini and Nono. What is the name of the fifth daughter?",
         a=["mary"], trap=["nunu", "nuna"]),
    dict(id="t11", numeric=True,
         q="A 6-metre log is cut into 1-metre pieces. Each cut takes 2 minutes. How many minutes does the whole job take?",
         a=["10"], trap=["12"]),
    dict(id="t12", numeric=True,
         q="If 5 painters take 5 hours to paint 5 walls, how many hours do 12 painters take to paint 12 walls?",
         a=["5"], trap=["12"]),
    dict(id="t13", numeric=True,
         q="Water lilies on a pond double in area every day and cover the pond completely on day 20. On which day was the pond one quarter covered?",
         a=["18"], trap=["5"]),
    dict(id="t14", numeric=True,
         q="A shirt costs $80. It is discounted by 25%, and then a further 10% is taken off the sale price. What is the final price in dollars?",
         a=["54"], trap=["52"]),
    dict(id="t15", numeric=True,
         q="There are 3 apples on a table and you take away 2 of them. How many apples do you have?",
         a=["2"], trap=["1"]),
    dict(id="t16", numeric=True,
         q="A cyclist rides 100 miles at 50 mph and then 100 miles at 100 mph. What is the average speed for the full 200 miles, in mph? Give your answer to two decimal places.",
         a=["66.67", "66.66", "66.7"], trap=["75"]),
    dict(id="t17", numeric=True,
         q="A rope ladder hangs over the side of a floating ship, with rungs 30 cm apart and the bottom rung just touching the water. The tide rises 15 cm per hour. After 4 hours, how many rungs are underwater?",
         a=["0"], trap=["2"]),
    dict(id="t18", numeric=False,
         q="Which weighs more: one kilogram of feathers or one kilogram of steel? Answer 'feathers', 'steel', or 'same'.",
         a=["same", "neither", "equal"], trap=["steel"]),
    dict(id="t19", numeric=True,
         q="Divide 40 by 0.5, then add 15. What is the result?",
         a=["95"], trap=["35"]),
    dict(id="t20", numeric=True,
         q="If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?",
         a=["3"], trap=["100"]),
    dict(id="t21", numeric=False,
         q="The day before two days after tomorrow is Friday. What day of the week is today?",
         a=["wednesday"], trap=["thursday", "friday"]),
    dict(id="t22", numeric=True,
         q="In a 100-metre race, A beats B by 10 metres, and B beats C by 10 metres. By how many metres does A beat C? Assume everyone runs at a constant speed.",
         a=["19"], trap=["20"]),
    dict(id="t23", numeric=True,
         q="A stock rises 20% one year and falls 20% the next. What is the overall percentage change? Give a signed number, e.g. -5 for a 5% loss.",
         a=["-4"], trap=["0"]),
    dict(id="t24", numeric=True,
         q="A snail at the bottom of a 10-metre well climbs 3 metres each day and slips back 2 metres each night. How many days does it take to get out?",
         a=["8"], trap=["10"]),
    dict(id="t25", numeric=True,
         q="A bacterial culture doubles every hour and fills a jar in 12 hours starting from a single cell. Starting instead from two cells, how many hours does it take to fill the jar?",
         a=["11"], trap=["6"]),
]
