from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

OP_EQ = 0
OP_LE = 1
OP_GE = 2
OP_RANGE = 3

OP_NAMES = {
    OP_EQ: "EQ",
    OP_LE: "LE",
    OP_GE: "GE",
    OP_RANGE: "RANGE",
}


@dataclass
class QueryCatalogue:
    m: int
    max_terms: int
    attrs: np.ndarray
    ops: np.ndarray
    values: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    num_terms: np.ndarray
    names: list[str]
    groups: list[str]
    families: list[str]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.attrs, self.ops, self.values, self.lows, self.highs

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "max_terms": self.max_terms,
            "attrs": self.attrs.tolist(),
            "ops": self.ops.tolist(),
            "values": self.values.tolist(),
            "lows": self.lows.tolist(),
            "highs": self.highs.tolist(),
            "num_terms": self.num_terms.tolist(),
            "names": self.names,
            "groups": self.groups,
            "families": self.families,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryCatalogue":
        return cls(
            m=int(data["m"]),
            max_terms=int(data["max_terms"]),
            attrs=np.asarray(data["attrs"], dtype=np.int32),
            ops=np.asarray(data["ops"], dtype=np.int32),
            values=np.asarray(data["values"], dtype=np.int32),
            lows=np.asarray(data["lows"], dtype=np.int32),
            highs=np.asarray(data["highs"], dtype=np.int32),
            num_terms=np.asarray(data["num_terms"], dtype=np.int32),
            names=list(data["names"]),
            groups=list(data["groups"]),
            families=list(data["families"]),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))

    def eval_query_np(self, X: np.ndarray, qid: int) -> np.ndarray:
        sat = np.ones(X.shape[0], dtype=bool)
        for t in range(int(self.num_terms[qid])):
            attr = int(self.attrs[qid, t])
            op = int(self.ops[qid, t])
            x = X[:, attr]
            if op == OP_EQ:
                cond = x == int(self.values[qid, t])
            elif op == OP_LE:
                cond = x <= int(self.values[qid, t])
            elif op == OP_GE:
                cond = x >= int(self.values[qid, t])
            elif op == OP_RANGE:
                cond = (x >= int(self.lows[qid, t])) & (x <= int(self.highs[qid, t]))
            else:
                raise ValueError(f"Unknown op {op}")
            sat &= cond
        return sat

    def query_terms(self, qid: int) -> list[tuple[int, int, int, int, int]]:
        terms: list[tuple[int, int, int, int, int]] = []
        for t in range(int(self.num_terms[qid])):
            terms.append(
                (
                    int(self.attrs[qid, t]),
                    int(self.ops[qid, t]),
                    int(self.values[qid, t]),
                    int(self.lows[qid, t]),
                    int(self.highs[qid, t]),
                )
            )
        return terms


def query_key(qcat: QueryCatalogue, qid: int) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(sorted(qcat.query_terms(qid)))


def filter_query_catalogue(qcat: QueryCatalogue, keep_indices: np.ndarray) -> QueryCatalogue:
    keep = np.asarray(keep_indices, dtype=np.int32)
    return QueryCatalogue(
        m=int(len(keep)),
        max_terms=int(qcat.max_terms),
        attrs=qcat.attrs[keep].copy(),
        ops=qcat.ops[keep].copy(),
        values=qcat.values[keep].copy(),
        lows=qcat.lows[keep].copy(),
        highs=qcat.highs[keep].copy(),
        num_terms=qcat.num_terms[keep].copy(),
        names=[qcat.names[int(i)] for i in keep],
        groups=[qcat.groups[int(i)] for i in keep],
        families=[qcat.families[int(i)] for i in keep],
    )


class QueryBuilder:
    def __init__(self, max_terms: int):
        self.max_terms = int(max_terms)
        self._attrs: list[list[int]] = []
        self._ops: list[list[int]] = []
        self._values: list[list[int]] = []
        self._lows: list[list[int]] = []
        self._highs: list[list[int]] = []
        self.names: list[str] = []
        self.groups: list[str] = []
        self.families: list[str] = []
        self._seen: set[tuple] = set()

    def add(
        self,
        terms: list[tuple[int, int, int, int, int]],
        name: str,
        group: str,
        family: str,
    ) -> bool:
        if not terms or len(terms) > self.max_terms:
            return False
        key = tuple(sorted(terms))
        if key in self._seen:
            return False
        self._seen.add(key)
        attrs = [-1] * self.max_terms
        ops = [OP_EQ] * self.max_terms
        values = [0] * self.max_terms
        lows = [0] * self.max_terms
        highs = [0] * self.max_terms
        for idx, (attr, op, value, lo, hi) in enumerate(terms):
            attrs[idx] = int(attr)
            ops[idx] = int(op)
            values[idx] = int(value)
            lows[idx] = int(lo)
            highs[idx] = int(hi)
        self._attrs.append(attrs)
        self._ops.append(ops)
        self._values.append(values)
        self._lows.append(lows)
        self._highs.append(highs)
        self.names.append(name)
        self.groups.append(group)
        self.families.append(family)
        return True

    def build(self) -> QueryCatalogue:
        m = len(self.names)
        return QueryCatalogue(
            m=m,
            max_terms=self.max_terms,
            attrs=np.asarray(self._attrs, dtype=np.int32).reshape(m, self.max_terms),
            ops=np.asarray(self._ops, dtype=np.int32).reshape(m, self.max_terms),
            values=np.asarray(self._values, dtype=np.int32).reshape(m, self.max_terms),
            lows=np.asarray(self._lows, dtype=np.int32).reshape(m, self.max_terms),
            highs=np.asarray(self._highs, dtype=np.int32).reshape(m, self.max_terms),
            num_terms=np.asarray([sum(1 for a in row if a >= 0) for row in self._attrs], dtype=np.int32),
            names=self.names,
            groups=self.groups,
            families=self.families,
        )
