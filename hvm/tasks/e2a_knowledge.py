"""E2a -- scenario X: knowledge-gated tasks.

Closed-book factual recall. The answer is in the weights or it is not, and no
loop retrieves a fact the model never learned. This is the arm where we expect
delta_harness ~= 0 and delta_model large and monotonic in scale.

Selection rules (to keep the arm honest):
  * Nothing time-sensitive -- no "current" officeholders, records or rankings,
    so the item set does not rot and knowledge cutoffs do not confound scale.
  * Nothing ambiguous -- exactly one defensible answer (this is why e.g.
    "which country uses the kwacha" was cut: Zambia and Malawi both do).
  * Graded obscurity -- some items a 1.5B model plainly knows, some it plainly
    does not, so the model axis has room to move.
"""
from __future__ import annotations

from typing import Any

# (question, accepted answers -- first is canonical)
E2A_SPECS: list[dict[str, Any]] = [
    dict(id="k01", q="Which chemical element has atomic number 42?",
         a=["molybdenum"]),
    dict(id="k02", q="In which year was the Peace of Westphalia signed?",
         a=["1648"]),
    dict(id="k03", q="What is the SI unit of electrical conductance?",
         a=["siemens"]),
    dict(id="k04", q="Who wrote the novel 'The Leopard' (Il Gattopardo)?",
         a=["giuseppe tomasi di lampedusa", "tomasi di lampedusa", "lampedusa"]),
    dict(id="k05", q="What is the capital city of Kyrgyzstan?",
         a=["bishkek"]),
    dict(id="k06", q="Which planet is orbited by the moon Titan?",
         a=["saturn"]),
    dict(id="k07", q="What is the longest bone in the human body?",
         a=["femur", "thigh bone"]),
    dict(id="k08", q="Who developed the first successful polio vaccine?",
         a=["jonas salk", "salk"]),
    dict(id="k09", q="What is the chemical formula of table salt?",
         a=["nacl"]),
    dict(id="k10", q="In which year did the Chernobyl nuclear disaster occur?",
         a=["1986"]),
    dict(id="k11", q="What is the longest river in Asia?",
         a=["yangtze", "yangtze river", "chang jiang"]),
    dict(id="k12", q="Which mathematician proved Fermat's Last Theorem?",
         a=["andrew wiles", "wiles"]),
    dict(id="k13", q="Which country's currency is the forint?",
         a=["hungary"]),
    dict(id="k14", q="What is the smallest prime number greater than 100?",
         a=["101"]),
    dict(id="k15", q="Who painted 'The Garden of Earthly Delights'?",
         a=["hieronymus bosch", "bosch"]),
    dict(id="k16", q="What is the biological process by which plants release water vapour through their leaves?",
         a=["transpiration"]),
    dict(id="k17", q="Which gas makes up roughly 78 percent of Earth's atmosphere?",
         a=["nitrogen", "n2"]),
    dict(id="k18", q="In which year was the Rosetta Stone discovered?",
         a=["1799"]),
    dict(id="k19", q="What is the capital city of Mongolia?",
         a=["ulaanbaatar", "ulan bator"]),
    dict(id="k20", q="Which physicist formulated the uncertainty principle?",
         a=["werner heisenberg", "heisenberg"]),
    dict(id="k21", q="What is the scientific study of fungi called?",
         a=["mycology"]),
    dict(id="k22", q="What is the deepest oceanic trench on Earth?",
         a=["mariana trench", "marianas trench"]),
    dict(id="k23", q="Who composed the opera 'The Magic Flute'?",
         a=["wolfgang amadeus mozart", "mozart"]),
    dict(id="k24", q="What is the chemical symbol for tungsten?",
         a=["w"]),
    dict(id="k25", q="Which war was ended by the Treaty of Ghent?",
         a=["war of 1812", "the war of 1812"]),
    dict(id="k26", q="What is the largest island in the Mediterranean Sea?",
         a=["sicily", "sicilia"]),
    dict(id="k27", q="Which blood type is known as the universal red-cell donor?",
         a=["o negative", "o-", "type o negative"]),
    dict(id="k28", q="What does the abbreviation SQL originally stand for?",
         a=["structured query language"]),
    dict(id="k29", q="Which vitamin does human skin synthesise when exposed to sunlight?",
         a=["vitamin d", "d"]),
    dict(id="k30", q="What is the currency of Vietnam?",
         a=["dong", "vietnamese dong"]),
    dict(id="k31", q="Who was the first woman to be awarded a Nobel Prize?",
         a=["marie curie", "curie", "marie sklodowska curie"]),
    dict(id="k32", q="Which large spiral galaxy is the nearest to the Milky Way?",
         a=["andromeda", "andromeda galaxy", "m31"]),
    dict(id="k33", q="What is the freezing point of water in kelvin?",
         a=["273.15", "273.15 k", "273"]),
    dict(id="k34", q="Which organ of the human body produces insulin?",
         a=["pancreas"]),
    dict(id="k35", q="Who discovered penicillin?",
         a=["alexander fleming", "fleming"]),
    dict(id="k36", q="What is the capital city of Ecuador?",
         a=["quito"]),
    dict(id="k37", q="In Greek mythology, who is the god of the forge and metalworking?",
         a=["hephaestus", "hephaistos"]),
    dict(id="k38", q="What is the chemical symbol for potassium?",
         a=["k"]),
    dict(id="k39", q="What is the highest mountain in Africa?",
         a=["kilimanjaro", "mount kilimanjaro", "mt kilimanjaro"]),
    dict(id="k40", q="Who wrote the novel 'One Hundred Years of Solitude'?",
         a=["gabriel garcia marquez", "garcia marquez", "marquez"]),
]
