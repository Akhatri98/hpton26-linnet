"""Element <-> salt matching for group memberships.  [deterministic]

Labels name the active part ("products containing calcium, iron, or zinc");
products are filed by the salt ("calcium_carbonate"). This table says which
salts carry which element. It is hand-written, small, and reviewable: it is
chemistry, not label reading, so no model is involved.

Used in two places:
  - extraction check: a proposed member "calcium_carbonate" is accepted against
    a sentence that only says "calcium", with provenance forced to "interpreted"
    (the label did not name the salt; we inferred it).
  - group expansion: a member that is an element ("calcium") covers every
    library ingredient carrying that element (calcium carbonate, calcium citrate...).
"""

from __future__ import annotations

import re

# element -> words that identify it at the START of a salt's name
ELEMENTS: dict[str, tuple[str, ...]] = {
    "calcium": ("calcium",),
    "magnesium": ("magnesium",),
    "aluminum": ("aluminum", "aluminium"),
    "iron": ("iron", "ferrous", "ferric"),
    "zinc": ("zinc",),
    "lanthanum": ("lanthanum",),
    "bismuth": ("bismuth",),
}


def element_of(ingredient: str) -> str | None:
    """'calcium_carbonate' -> 'calcium'; 'ferrous_sulfate' -> 'iron'; 'colestipol' -> None."""
    first = ingredient.lower().replace("_", " ").split(" ")[0]
    for el, words in ELEMENTS.items():
        if first in words:
            return el
    return None


def is_element(name: str) -> bool:
    return name.lower() in ELEMENTS


def named_in(ingredient: str, text: str) -> str | None:
    """How a sentence names an ingredient:
    'exact'   - the ingredient itself ('calcium carbonate', or element 'calcium');
    'element' - only its element ('calcium' for calcium_carbonate; 'iron' for ferrous_sulfate);
    None      - not at all."""
    low = text.lower()
    if re.search(rf"\b{re.escape(ingredient.lower().replace('_', ' '))}\b", low):
        return "exact"
    el = element_of(ingredient)
    if el and any(re.search(rf"\b{w}\b", low) for w in (el, *ELEMENTS[el])):
        return "element"
    return None
