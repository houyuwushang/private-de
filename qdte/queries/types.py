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
    linear_attrs: np.ndarray
    linear_weights: np.ndarray
    linear_thresholds: np.ndarray
    linear_num_terms: np.ndarray
    names: list[str]
    groups: list[str]
    families: list[str]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.attrs, self.ops, self.values, self.lows, self.highs

    def eval_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.attrs,
            self.ops,
            self.values,
            self.lows,
            self.highs,
            self.linear_attrs,
            self.linear_weights,
            self.linear_thresholds,
            self.linear_num_terms,
        )

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
            "linear_attrs": self.linear_attrs.tolist(),
            "linear_weights": self.linear_weights.tolist(),
            "linear_thresholds": self.linear_thresholds.tolist(),
            "linear_num_terms": self.linear_num_terms.tolist(),
            "names": self.names,
            "groups": self.groups,
            "families": self.families,
        }

    def validate(self, cardinalities: np.ndarray | None = None) -> None:
        if int(self.m) < 0:
            raise ValueError("Query catalogue size must be non-negative")
        if int(self.max_terms) <= 0:
            raise ValueError("Query catalogue max_terms must be positive")
        matrix_shape = (int(self.m), int(self.max_terms))
        for name, values in (
            ("attrs", self.attrs),
            ("ops", self.ops),
            ("values", self.values),
            ("lows", self.lows),
            ("highs", self.highs),
            ("linear_attrs", self.linear_attrs),
            ("linear_weights", self.linear_weights),
        ):
            if np.asarray(values).shape != matrix_shape:
                raise ValueError(f"Query catalogue {name} must have shape {matrix_shape}")
        for name, values in (
            ("num_terms", self.num_terms),
            ("linear_thresholds", self.linear_thresholds),
            ("linear_num_terms", self.linear_num_terms),
        ):
            if np.asarray(values).shape != (int(self.m),):
                raise ValueError(f"Query catalogue {name} must have shape ({int(self.m)},)")
        for name, values in (("names", self.names), ("groups", self.groups), ("families", self.families)):
            if len(values) != int(self.m):
                raise ValueError(f"Query catalogue {name} must contain {int(self.m)} entries")
        if not np.all(np.isfinite(np.asarray(self.linear_weights, dtype=np.float64))):
            raise ValueError("Query catalogue linear weights must be finite")
        if not np.all(np.isfinite(np.asarray(self.linear_thresholds, dtype=np.float64))):
            raise ValueError("Query catalogue linear thresholds must be finite")

        cards: np.ndarray | None = None
        if cardinalities is not None:
            cards = np.asarray(cardinalities, dtype=np.int64)
            if cards.ndim != 1 or cards.size == 0 or np.any(cards <= 0):
                raise ValueError("Public query cardinalities must be a non-empty positive 1D vector")

        valid_ops = {OP_EQ, OP_LE, OP_GE, OP_RANGE}
        for qid in range(int(self.m)):
            num_terms = int(self.num_terms[qid])
            linear_num_terms = int(self.linear_num_terms[qid])
            if not 0 <= num_terms <= int(self.max_terms):
                raise ValueError(f"Query {qid} has invalid num_terms={num_terms}")
            if not 0 <= linear_num_terms <= int(self.max_terms):
                raise ValueError(f"Query {qid} has invalid linear_num_terms={linear_num_terms}")
            if num_terms + linear_num_terms == 0:
                raise ValueError(f"Query {qid} has no active terms")
            if num_terms > 0 and linear_num_terms > 0:
                raise ValueError(f"Query {qid} mixes ordinary and linear terms, which is not supported")

            ordinary_attrs = np.asarray(self.attrs[qid], dtype=np.int64)
            if np.any(ordinary_attrs[:num_terms] < 0) or np.any(ordinary_attrs[num_terms:] != -1):
                raise ValueError(f"Query {qid} ordinary terms are not canonically packed")
            if len(np.unique(ordinary_attrs[:num_terms])) != num_terms:
                raise ValueError(f"Query {qid} contains duplicate ordinary attributes")
            for term in range(num_terms):
                attr = int(ordinary_attrs[term])
                op = int(self.ops[qid, term])
                if op not in valid_ops:
                    raise ValueError(f"Query {qid} has unknown op {op}")
                if cards is not None:
                    if attr >= len(cards):
                        raise ValueError(f"Query {qid} references out-of-range attribute {attr}")
                    cardinality = int(cards[attr])
                    if op == OP_RANGE:
                        lo = int(self.lows[qid, term])
                        hi = int(self.highs[qid, term])
                        if not 0 <= lo <= hi < cardinality:
                            raise ValueError(f"Query {qid} has range [{lo}, {hi}] outside attribute {attr}")
                    else:
                        value = int(self.values[qid, term])
                        if not 0 <= value < cardinality:
                            raise ValueError(f"Query {qid} has value {value} outside attribute {attr}")

            linear_attrs = np.asarray(self.linear_attrs[qid], dtype=np.int64)
            if np.any(linear_attrs[:linear_num_terms] < 0) or np.any(linear_attrs[linear_num_terms:] != -1):
                raise ValueError(f"Query {qid} linear terms are not canonically packed")
            if len(np.unique(linear_attrs[:linear_num_terms])) != linear_num_terms:
                raise ValueError(f"Query {qid} contains duplicate linear attributes")
            if cards is not None and np.any(linear_attrs[:linear_num_terms] >= len(cards)):
                raise ValueError(f"Query {qid} references an out-of-range linear attribute")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryCatalogue":
        catalogue = cls(
            m=int(data["m"]),
            max_terms=int(data["max_terms"]),
            attrs=np.asarray(data["attrs"], dtype=np.int32),
            ops=np.asarray(data["ops"], dtype=np.int32),
            values=np.asarray(data["values"], dtype=np.int32),
            lows=np.asarray(data["lows"], dtype=np.int32),
            highs=np.asarray(data["highs"], dtype=np.int32),
            num_terms=np.asarray(data["num_terms"], dtype=np.int32),
            linear_attrs=np.asarray(data.get("linear_attrs", [[-1] * int(data["max_terms"])] * int(data["m"])), dtype=np.int32),
            linear_weights=np.asarray(
                data.get("linear_weights", [[0.0] * int(data["max_terms"])] * int(data["m"])), dtype=np.float32
            ),
            linear_thresholds=np.asarray(data.get("linear_thresholds", [0.0] * int(data["m"])), dtype=np.float32),
            linear_num_terms=np.asarray(data.get("linear_num_terms", [0] * int(data["m"])), dtype=np.int32),
            names=list(data["names"]),
            groups=list(data["groups"]),
            families=list(data["families"]),
        )
        catalogue.validate()
        return catalogue

    def save_json(self, path: str | Path) -> None:
        self.validate()
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
        if int(self.linear_num_terms[qid]) > 0:
            score = np.zeros(X.shape[0], dtype=np.float32)
            for t in range(int(self.linear_num_terms[qid])):
                attr = int(self.linear_attrs[qid, t])
                weight = float(self.linear_weights[qid, t])
                score += weight * X[:, attr].astype(np.float32)
            sat &= score <= float(self.linear_thresholds[qid])
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

    def linear_terms(self, qid: int) -> list[tuple[int, float]]:
        terms: list[tuple[int, float]] = []
        for t in range(int(self.linear_num_terms[qid])):
            terms.append((int(self.linear_attrs[qid, t]), float(self.linear_weights[qid, t])))
        return terms


def query_key(qcat: QueryCatalogue, qid: int) -> tuple[tuple[int, int, int, int, int], ...]:
    ordinary = tuple(sorted(qcat.query_terms(qid)))
    linear = tuple(sorted((attr, round(weight, 8)) for attr, weight in qcat.linear_terms(qid)))
    if linear:
        return ordinary + (("__halfspace__", linear, round(float(qcat.linear_thresholds[qid]), 8)),)  # type: ignore[return-value]
    return ordinary


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
        linear_attrs=qcat.linear_attrs[keep].copy(),
        linear_weights=qcat.linear_weights[keep].copy(),
        linear_thresholds=qcat.linear_thresholds[keep].copy(),
        linear_num_terms=qcat.linear_num_terms[keep].copy(),
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
        self._linear_attrs: list[list[int]] = []
        self._linear_weights: list[list[float]] = []
        self._linear_thresholds: list[float] = []
        self._linear_num_terms: list[int] = []
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
        self._linear_attrs.append([-1] * self.max_terms)
        self._linear_weights.append([0.0] * self.max_terms)
        self._linear_thresholds.append(0.0)
        self._linear_num_terms.append(0)
        self.names.append(name)
        self.groups.append(group)
        self.families.append(family)
        return True

    def add_halfspace(
        self,
        linear_terms: list[tuple[int, float]],
        threshold: float,
        name: str,
        group: str,
        family: str = "halfspace",
    ) -> bool:
        if not linear_terms or len(linear_terms) > self.max_terms:
            return False
        merged: dict[int, float] = {}
        for attr, weight in linear_terms:
            merged[int(attr)] = merged.get(int(attr), 0.0) + float(weight)
        terms = tuple(sorted((attr, round(weight, 8)) for attr, weight in merged.items() if abs(weight) > 1.0e-12))
        if not terms:
            return False
        key = ("halfspace", terms, round(float(threshold), 8))
        if key in self._seen:
            return False
        self._seen.add(key)
        self._attrs.append([-1] * self.max_terms)
        self._ops.append([OP_EQ] * self.max_terms)
        self._values.append([0] * self.max_terms)
        self._lows.append([0] * self.max_terms)
        self._highs.append([0] * self.max_terms)
        linear_attrs = [-1] * self.max_terms
        linear_weights = [0.0] * self.max_terms
        for idx, (attr, weight) in enumerate(terms):
            linear_attrs[idx] = int(attr)
            linear_weights[idx] = float(weight)
        self._linear_attrs.append(linear_attrs)
        self._linear_weights.append(linear_weights)
        self._linear_thresholds.append(float(threshold))
        self._linear_num_terms.append(len(terms))
        self.names.append(name)
        self.groups.append(group)
        self.families.append(family)
        return True

    def build(self) -> QueryCatalogue:
        m = len(self.names)
        catalogue = QueryCatalogue(
            m=m,
            max_terms=self.max_terms,
            attrs=np.asarray(self._attrs, dtype=np.int32).reshape(m, self.max_terms),
            ops=np.asarray(self._ops, dtype=np.int32).reshape(m, self.max_terms),
            values=np.asarray(self._values, dtype=np.int32).reshape(m, self.max_terms),
            lows=np.asarray(self._lows, dtype=np.int32).reshape(m, self.max_terms),
            highs=np.asarray(self._highs, dtype=np.int32).reshape(m, self.max_terms),
            num_terms=np.asarray([sum(1 for a in row if a >= 0) for row in self._attrs], dtype=np.int32),
            linear_attrs=np.asarray(self._linear_attrs, dtype=np.int32).reshape(m, self.max_terms),
            linear_weights=np.asarray(self._linear_weights, dtype=np.float32).reshape(m, self.max_terms),
            linear_thresholds=np.asarray(self._linear_thresholds, dtype=np.float32).reshape(m),
            linear_num_terms=np.asarray(self._linear_num_terms, dtype=np.int32).reshape(m),
            names=self.names,
            groups=self.groups,
            families=self.families,
        )
        catalogue.validate()
        return catalogue
