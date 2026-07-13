from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qdte.queries.types import OP_EQ, OP_GE, OP_LE, OP_RANGE, QueryCatalogue


@dataclass(frozen=True)
class SparseCandidateDelta:
    qids: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class QueryDeltaIndex:
    qcat: QueryCatalogue
    attr_to_qids: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, qcat: QueryCatalogue, num_attrs: int | None = None) -> "QueryDeltaIndex":
        max_attr = -1
        ordinary = qcat.attrs[qcat.attrs >= 0]
        linear = qcat.linear_attrs[qcat.linear_attrs >= 0]
        if ordinary.size:
            max_attr = max(max_attr, int(ordinary.max()))
        if linear.size:
            max_attr = max(max_attr, int(linear.max()))
        width = max(max_attr + 1, int(num_attrs or 0))
        attr_lists: list[list[int]] = [[] for _ in range(width)]
        for qid in range(qcat.m):
            attrs: set[int] = set()
            for attr, *_ in qcat.query_terms(qid):
                attrs.add(int(attr))
            for attr, _ in qcat.linear_terms(qid):
                attrs.add(int(attr))
            for attr in attrs:
                if attr < 0:
                    continue
                while attr >= len(attr_lists):
                    attr_lists.append([])
                attr_lists[attr].append(qid)
        return cls(
            qcat=qcat,
            attr_to_qids=tuple(np.asarray(sorted(set(qids)), dtype=np.int32) for qids in attr_lists),
        )

    def affected_query_ids(self, old_row: np.ndarray, new_row: np.ndarray) -> np.ndarray:
        changed_attrs = np.flatnonzero(np.asarray(old_row) != np.asarray(new_row))
        if len(changed_attrs) == 0:
            return np.empty(0, dtype=np.int32)
        parts: list[np.ndarray] = []
        for attr in changed_attrs.tolist():
            if 0 <= int(attr) < len(self.attr_to_qids):
                qids = self.attr_to_qids[int(attr)]
                if len(qids):
                    parts.append(qids)
        if not parts:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)

    def candidate_delta(self, old_row: np.ndarray, new_row: np.ndarray) -> SparseCandidateDelta:
        qids = self.affected_query_ids(old_row, new_row)
        if len(qids) == 0:
            return SparseCandidateDelta(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8))
        old_sat = self._eval_queries_on_row(old_row, qids)
        new_sat = self._eval_queries_on_row(new_row, qids)
        values = new_sat.astype(np.int8) - old_sat.astype(np.int8)
        keep = values != 0
        return SparseCandidateDelta(
            qids=qids[keep].astype(np.int32, copy=False),
            values=values[keep].astype(np.int8, copy=False),
        )

    def dense_candidate_deltas(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        if old.shape != new.shape:
            raise ValueError(f"old_rows and new_rows must have the same shape, got {old.shape} and {new.shape}")
        out = np.zeros((old.shape[0], self.qcat.m), dtype=np.int8)
        for idx in range(old.shape[0]):
            delta = self.candidate_delta(old[idx], new[idx])
            out[idx, delta.qids] = delta.values
        return out

    def delta_sum(self, old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        sparse = self.sparse_delta_sum(old_rows, new_rows)
        total = np.zeros(self.qcat.m, dtype=np.float32)
        total[sparse.qids] = sparse.values.astype(np.float32)
        return total

    def sparse_delta_sum(self, old_rows: np.ndarray, new_rows: np.ndarray) -> SparseCandidateDelta:
        old = np.asarray(old_rows, dtype=np.int32)
        new = np.asarray(new_rows, dtype=np.int32)
        if old.shape != new.shape:
            raise ValueError(f"old_rows and new_rows must have the same shape, got {old.shape} and {new.shape}")
        qid_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for idx in range(old.shape[0]):
            delta = self.candidate_delta(old[idx], new[idx])
            if len(delta.qids):
                qid_parts.append(delta.qids)
                value_parts.append(delta.values)
        if not qid_parts:
            return SparseCandidateDelta(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int8),
            )
        if len(qid_parts) == 1:
            return SparseCandidateDelta(
                qid_parts[0].astype(np.int32, copy=False),
                value_parts[0].astype(np.int8, copy=False),
            )

        qids = np.concatenate(qid_parts)
        values = np.concatenate(value_parts).astype(np.int16, copy=False)
        order = np.argsort(qids, kind="stable")
        sorted_qids = qids[order]
        sorted_values = values[order]
        starts = np.flatnonzero(
            np.concatenate(
                [np.asarray([True]), sorted_qids[1:] != sorted_qids[:-1]]
            )
        )
        summed_values = np.add.reduceat(sorted_values, starts)
        keep = summed_values != 0
        return SparseCandidateDelta(
            sorted_qids[starts][keep].astype(np.int32, copy=False),
            summed_values[keep].astype(np.int8, copy=False),
        )

    def padded_attr_query_ids(
        self,
        num_attrs: int | None = None,
        pad_value: int | None = None,
    ) -> np.ndarray:
        width = max(len(self.attr_to_qids), int(num_attrs or 0))
        pad = int(self.qcat.m if pad_value is None else pad_value)
        max_len = max((len(qids) for qids in self.attr_to_qids), default=0)
        out = np.full((width, max_len), pad, dtype=np.int32)
        for attr, qids in enumerate(self.attr_to_qids):
            if len(qids):
                out[attr, : len(qids)] = qids
        return out

    def query_attr_matrix(self, num_attrs: int | None = None) -> np.ndarray:
        width = max(len(self.attr_to_qids), int(num_attrs or 0))
        out = np.zeros((self.qcat.m, width), dtype=bool)
        for qid in range(self.qcat.m):
            for attr, *_ in self.qcat.query_terms(qid):
                if 0 <= int(attr) < width:
                    out[qid, int(attr)] = True
            for attr, _ in self.qcat.linear_terms(qid):
                if 0 <= int(attr) < width:
                    out[qid, int(attr)] = True
        return out

    def query_attr_bits(self, num_attrs: int | None = None) -> np.ndarray:
        words = self.query_attr_words(num_attrs=num_attrs)
        if words.shape[1] > 1:
            raise ValueError("query_attr_bits is a single-word helper; use query_attr_words for more than 32 attributes")
        return words[:, 0]

    def query_attr_words(self, num_attrs: int | None = None) -> np.ndarray:
        width = max(len(self.attr_to_qids), int(num_attrs or 0))
        word_count = max(1, (width + 31) // 32)
        out = np.zeros((self.qcat.m, word_count), dtype=np.uint32)
        matrix = self.query_attr_matrix(num_attrs=width)
        for attr in range(width):
            word = attr // 32
            offset = attr % 32
            out[:, word] = out[:, word] | (matrix[:, attr].astype(np.uint32) << np.uint32(offset))
        return out

    def _eval_query_on_row(self, row: np.ndarray, qid: int) -> bool:
        qcat = self.qcat
        for attr, op, value, lo, hi in qcat.query_terms(qid):
            x = int(row[int(attr)])
            if op == OP_EQ:
                ok = x == int(value)
            elif op == OP_LE:
                ok = x <= int(value)
            elif op == OP_GE:
                ok = x >= int(value)
            elif op == OP_RANGE:
                ok = int(lo) <= x <= int(hi)
            else:
                raise ValueError(f"Unknown op {op}")
            if not ok:
                return False
        if int(qcat.linear_num_terms[qid]) > 0:
            score = 0.0
            for attr, weight in qcat.linear_terms(qid):
                score += float(weight) * float(row[int(attr)])
            if score > float(qcat.linear_thresholds[qid]):
                return False
        return True

    def _eval_queries_on_row(self, row: np.ndarray, qids: np.ndarray) -> np.ndarray:
        qcat = self.qcat
        ids = np.asarray(qids, dtype=np.int32)
        if len(ids) == 0:
            return np.empty(0, dtype=np.bool_)
        row_arr = np.asarray(row, dtype=np.int32)

        attrs = qcat.attrs[ids]
        valid = attrs >= 0
        xvals = row_arr[np.maximum(attrs, 0)]
        ops = qcat.ops[ids]
        values = qcat.values[ids]
        cond = np.where(ops == OP_EQ, xvals == values, xvals <= values)
        cond = np.where(ops == OP_GE, xvals >= values, cond)
        cond = np.where(ops == OP_RANGE, (xvals >= qcat.lows[ids]) & (xvals <= qcat.highs[ids]), cond)
        ordinary_sat = np.all(np.where(valid, cond, True), axis=1)

        linear_attrs = qcat.linear_attrs[ids]
        linear_valid_terms = linear_attrs >= 0
        linear_xvals = row_arr[np.maximum(linear_attrs, 0)].astype(np.float32)
        linear_scores = np.sum(
            np.where(linear_valid_terms, linear_xvals * qcat.linear_weights[ids], 0.0),
            axis=1,
            dtype=np.float32,
        )
        has_linear = qcat.linear_num_terms[ids] > 0
        linear_sat = np.where(has_linear, linear_scores <= qcat.linear_thresholds[ids], True)
        return ordinary_sat & linear_sat
