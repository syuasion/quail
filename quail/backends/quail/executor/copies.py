"""Rows whose prompt tokens repeat an earlier row's, asked once per group."""

import numpy as np

from quail.catalog.tokens import first_copies


class Copies:
    """Rows of one run grouped by identical prompt tokens.

    The run asks the first row of each group; the other rows take its
    answers. A row keeps its own id in every result.

    Attributes:
        ids: Every row's document id, in input order.
        asked: Per row, the position of its group's first row among the
            asked rows.
        asked_rows: The asked rows' positions, in input order.
        asked_ids: The asked rows' document ids, in input order.
    """

    def __init__(self, ids, first):
        self.ids = np.asarray(ids)
        self.asked_rows = np.flatnonzero(first == np.arange(len(first)))
        self.asked = np.searchsorted(self.asked_rows, first)
        self.asked_ids = self.ids[self.asked_rows]
        order = np.argsort(self.asked, kind="stable")
        bounds = np.searchsorted(self.asked[order],
                                 np.arange(len(self.asked_rows) + 1))
        self._rows = [order[low:high]
                      for low, high in zip(bounds[:-1], bounds[1:])]
        self._index = {int(asked): index
                       for index, asked in enumerate(self.asked_ids.tolist())}

    @classmethod
    def of(cls, documents, ids) -> "Copies | None":
        """Return the copies among the documents, or None when no row repeats.

        Args:
            documents: Each row's prompt tokens before its first question.
            ids: Each row's document id, in the same order.
        """
        first = first_copies(documents)
        if (first == np.arange(len(first))).all():
            return None
        return cls(ids, first)

    def rows_of(self, index) -> np.ndarray:
        """Return the positions of the rows that asked row `index` answers for."""
        return self._rows[index]

    def ids_of(self, asked_id) -> np.ndarray:
        """Return the ids of the rows that the asked row `asked_id` answers for."""
        return self.ids[self._rows[self._index[int(asked_id)]]]
