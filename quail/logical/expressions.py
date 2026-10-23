"""Expressions used by logical query operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.predicates import (
    COLUMN_COMPARISONS,
    COMPARISON_KERNELS,
    TEXT_COMPARISONS,
    accepted_rows,
    compare_literal,
    membership,
)

DESCRIPTION_WORDS = 25
TASK_DESCRIPTION_WORDS = 50
# the functions a ScalarFunction may compute, by name
SCALAR_FUNCTIONS = {"length": pc.utf8_length, "lower": pc.utf8_lower,
                    "upper": pc.utf8_upper, "trim": pc.utf8_trim_whitespace}


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@dataclass(frozen=True)
class ColumnRef:
    alias: str  # table alias in the query ("r")
    provider: str  # provider name in the catalog ("reviews")
    column: str  # column name ("review")

    type_name: ClassVar[str] = "quail.column_ref"

    def validate(self) -> None:
        if not self.alias or not self.column:
            raise CompileError("a column reference names a table alias and a column")

    def __str__(self) -> str:
        return column_name(self)


def _validate_tree(expression) -> None:
    """Validate one value expression, whatever its node type."""
    if not isinstance(expression, (ColumnRef, GetField, ScalarFunction, Alias)):
        raise CompileError(
            f"a value expression is a column, a field, a function, or an "
            f"alias, got {type(expression).__name__}")
    expression.validate()


@dataclass(frozen=True)
class GetField:
    """One field of a struct-valued expression."""

    input: Expression
    field: str

    type_name: ClassVar[str] = "quail.get_field"

    @property
    def alias(self) -> str:
        return table_alias(self)

    def validate(self) -> None:
        _validate_tree(self.input)
        if not isinstance(self.field, str) or not self.field:
            raise CompileError("a field access names a struct field")

    def __str__(self) -> str:
        return column_name(self)


@dataclass(frozen=True)
class ScalarFunction:
    """A function of value expressions, computed row by row without the model."""

    function: str
    args: tuple[Expression, ...]

    type_name: ClassVar[str] = "quail.scalar_function"

    @property
    def alias(self) -> str:
        return table_alias(self)

    def validate(self) -> None:
        if self.function not in SCALAR_FUNCTIONS:
            raise CompileError(
                f"unsupported column function {self.function!r}; "
                f"supported: {', '.join(SCALAR_FUNCTIONS)}")
        for argument in self.args:
            _validate_tree(argument)

    def __str__(self) -> str:
        return column_name(self)


def column_expression(reference: ColumnRef, path: tuple = (),
                      function: Optional[str] = None, name: Optional[str] = None):
    """Build the expression tree a front end writes as fields, a function, and a name.

    For example, ``col("r.details").field("body").length().named("n")``
    becomes ``Alias(ScalarFunction("length", (GetField(ref, "body"),)), "n")``.
    """
    expression = reference
    for field in path:
        expression = GetField(expression, field)
    if function is not None:
        expression = ScalarFunction(function, (expression,))
    return expression if name is None else Alias(expression, name)


def evaluate_expression(expression, columns):
    """Evaluate a value expression over the columns of one table.

    Args:
        expression: A column name, or a tree of ColumnRef, GetField,
            ScalarFunction, and Alias nodes.
        columns: The table, record batch, or scan input the columns come
            from, read with ``column(name)``.

    Raises:
        CompileError: A field does not exist, or a function does not
            take its arguments' types.
    """
    if isinstance(expression, str):
        return columns.column(expression)
    if isinstance(expression, ColumnRef):
        return columns.column(expression.column)
    if isinstance(expression, Alias):
        return evaluate_expression(expression.expression, columns)
    if isinstance(expression, GetField):
        values = evaluate_expression(expression.input, columns)
        read = column_name(expression.input)
        if not pa.types.is_struct(values.type):
            raise CompileError(f"{read} has no fields; it is {values.type}")
        if values.type.get_field_index(expression.field) < 0:
            raise CompileError(
                f"{read} has no field {expression.field!r}; its fields are "
                f"{[item.name for item in values.type]}")
        return pc.struct_field(values, expression.field)
    arguments = [evaluate_expression(argument, columns) for argument in expression.args]
    try:
        return SCALAR_FUNCTIONS[expression.function](*arguments)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError,
            TypeError) as error:
        raise CompileError(
            f"{column_name(expression)} cannot be evaluated on "
            f"{[str(values.type) for values in arguments]}: {error}") from error


def column_references(expression) -> tuple[ColumnRef, ...]:
    """Return the source columns a value expression reads, in first use order."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, Alias):
        return column_references(expression.expression)
    if isinstance(expression, GetField):
        return column_references(expression.input)
    if isinstance(expression, ScalarFunction):
        return tuple(dict.fromkeys(
            ref for argument in expression.args for ref in column_references(argument)))
    return ()


def source_columns(expression) -> tuple[str, ...]:
    """Return the names of the source columns a document column reads."""
    if isinstance(expression, str):
        return (expression,)
    return tuple(dict.fromkeys(ref.column for ref in column_references(expression)))


def table_alias(expression) -> str:
    """Return the one table alias a value expression reads.

    Raises:
        CompileError: The expression reads two tables, or none.
    """
    aliases = tuple(dict.fromkeys(ref.alias for ref in column_references(expression)))
    if len(aliases) != 1:
        raise CompileError(
            f"{column_name(expression)} must read one table, not {list(aliases)}")
    return aliases[0]


LABEL_PREFIX = " "


def column_name(column, *, qualified: bool = True) -> str:
    """Return the name used for a source, computed, or projected column.

    Args:
        column: A column name, a value expression, or a projected or
            aggregated expression with a name.
        qualified: Whether a column reference carries its table alias.
    """
    if isinstance(column, str):
        return column
    if isinstance(column, ColumnRef):
        return f"{column.alias}.{column.column}" if qualified else column.column
    if isinstance(column, GetField):
        return f"{column_name(column.input, qualified=qualified)}.{column.field}"
    if isinstance(column, ScalarFunction):
        arguments = ", ".join(column_name(argument, qualified=qualified)
                              for argument in column.args)
        return f"{column.function}({arguments})"
    return column.name


def aliased_call(expression, *kinds):
    """Return the model call an Alias projects, or None.

    Args:
        expression: A projected column.
        kinds: When given, the model call kinds that count.
    """
    if not isinstance(expression, Alias) or not isinstance(
            expression.expression, ModelCall):
        return None
    call = expression.expression
    return call if not kinds or call.kind in kinds else None




@dataclass(frozen=True)
class Prompt:
    """The text of one model call, laid out for a model's prompt format.

    The planner lowers the text to token ids with the session's
    tokenizer; see quail.planner.prompt_lowering. A classification
    prompt keeps the text before each label in label_prefix and what
    rendered its label list, so lowering can render it again with one
    letter per label.

    Attributes:
        template: The template as the query names the predicate.
        args: Column references in placeholder order.
        preamble: The text before the document.
        tail: The text after the document, or after a join's last
            partner, with the document placeholder.
        frame: The text after the anchor document that every pair shares.
        frame_text: The text whose token count is the frame's; empty
            when the prompt has no frame.
        kind: filter, join, label, extract, or score.
        layout: The PROMPT_LAYOUTS name the text was built with.
        tail_segments: The tail after the placeholder, one entry per
            separately tokenized segment; empty means one segment.
        label_texts: Per placeholder of a join or a classification of
            joined rows, (alias, its label text, its frame text).
        label_prefix: The text before a scored letter or label.
        categories: A classification's labels, in query order.
        descriptions: Their descriptions, in label order.
        task_description: A classification's extra instructions.
        turn: The model's chat text around the prompt.
        source: The classification template as written.
        letters: The letters naming the labels of a lettered prompt.
    """

    template: str
    args: tuple  # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    frame: str = ""
    frame_text: str = ""
    kind: str = "filter"
    layout: str = "ai-if"
    tail_segments: tuple = ()
    label_texts: tuple = ()
    label_prefix: str = LABEL_PREFIX
    categories: tuple = ()
    descriptions: tuple = ()
    task_description: str = ""
    turn: tuple = ("", "")
    source: str = ""
    letters: tuple = ()

    type_name: ClassVar[str] = "quail.prompt"


DEFAULT_SELECTIVITY = 0.2


SCORE_COMPARISONS = ("<", "<=", ">", ">=")


MODEL_CALL_KINDS = ("boolean", "score", "label", "extract")


def effective_selectivity(selectivity: Optional[float]) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


PROBABILITIES_SUFFIX = "_probabilities"
# an "extract" call's second result column: the answer's character
# span in the document
SPAN_SUFFIX = "_span"


@dataclass(frozen=True)
class ModelCall:
    """One prompt asked of every row; the model answers with a value.

    The AI functions of the language are all model calls. They differ
    only in ``kind``, the type of the answer.
    """

    prompt: Prompt
    kind: str = "boolean"
    # a "label" call only: the labels in written order, and one
    # description per label, empty for none
    labels: tuple = ()
    descriptions: tuple = ()
    # a "label" call only: whether the result also carries each label's
    # probability, as the column named after the label column plus
    # PROBABILITIES_SUFFIX
    probabilities: bool = False
    # an "extract" call only: the question the answer is copied for
    question: str = ""

    type_name: ClassVar[str] = "quail.model_call"

    def aliases(self) -> tuple[str, ...]:
        """Return the table aliases the prompt reads, in placeholder order."""
        return tuple(dict.fromkeys(ref.alias for ref in self.prompt.args))

    def validate(self) -> None:
        if self.kind not in MODEL_CALL_KINDS:
            raise CompileError(
                f"model call kind must be one of {MODEL_CALL_KINDS}, got {self.kind!r}"
            )
        if self.kind != "label" and (
            self.labels or self.descriptions or self.probabilities
        ):
            raise CompileError("only an AI.CLASSIFY call has labels")
        if self.kind != "extract" and self.question:
            raise CompileError("only an AI.EXTRACT call has a question")
        if self.kind == "label":
            validate_labels(self.labels, self.descriptions)
        elif self.kind == "extract":
            validate_question(self.question)


def validate_question(question: str) -> None:
    """Check an AI.EXTRACT question: text with at least one word.

    Raises:
        CompileError: The question is empty or not a string.
    """
    if not isinstance(question, str) or not question.strip():
        raise CompileError("AI.EXTRACT needs a question with at least one word")


@dataclass(frozen=True)
class ExtractStruct:
    """Several AI.EXTRACT fields of one table, returned as struct columns.

    Each field is its own extraction, projected as an Alias named
    ``name.field``. The result packs the fields' answers into the
    struct column ``name`` and their spans into ``name`` plus
    SPAN_SUFFIX, each with one struct field per extraction field.

    Attributes:
        name: The struct column's name.
        fields: The field names, in written order.
    """

    name: str
    fields: tuple[str, ...]

    type_name: ClassVar[str] = "quail.extract_struct"

    def member(self, field: str) -> str:
        """Return the name of one field's extraction."""
        return f"{self.name}.{field}"

    @property
    def members(self) -> tuple[str, ...]:
        """Return the names of the fields' extractions, in field order."""
        return tuple(self.member(field) for field in self.fields)

    def packing(self) -> tuple:
        """Return the answer and span struct columns as Project packs them.

        Returns:
            (struct column, ((field, member column), ...)) for the
            answers, then for the spans.
        """
        return tuple(
            (self.name + suffix,
             tuple((field, self.member(field) + suffix) for field in self.fields))
            for suffix in ("", SPAN_SUFFIX))

    def validate(self) -> None:
        if not self.name or "." in self.name:
            raise CompileError(
                f"an extraction needs a column name without a dot, "
                f"got {self.name!r}")
        if not self.fields:
            raise CompileError(f"AI.EXTRACT {self.name!r} needs at least one field")
        for field in self.fields:
            if not isinstance(field, str) or not field or "." in field:
                raise CompileError(
                    f"an AI.EXTRACT field needs a name without a dot, "
                    f"got {field!r}")
        if len(set(self.fields)) != len(self.fields):
            raise CompileError(
                f"AI.EXTRACT {self.name!r} names a field twice: {list(self.fields)}")


def packed_columns(columns, packing) -> tuple[str, ...]:
    """Return result column names after packing members into struct columns.

    Each struct column takes the place of its first member; its other
    members leave the result.

    Args:
        columns: The column names before packing.
        packing: (struct column, ((field, member column), ...)) pairs.
    """
    owner = {member: name for name, fields in packing for _, member in fields}
    out = []
    for column in columns:
        name = owner.get(column, column)
        if name not in out:
            out.append(name)
    return tuple(out)


def validate_labels(labels: tuple, descriptions: tuple = ()) -> None:
    """Check an AI.CLASSIFY label list and its descriptions.

    Raises:
        CompileError: Fewer than two labels, an empty label, two labels
            equal ignoring case, or a description count that differs
            from the label count.
    """
    if len(labels) < 2:
        raise CompileError("AI.CLASSIFY needs at least two labels")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise CompileError("every AI.CLASSIFY label is non-empty text")
    folded = [label.casefold() for label in labels]
    if len(set(folded)) != len(folded):
        raise CompileError(
            f"AI.CLASSIFY labels must differ ignoring case, got {list(labels)}"
        )
    if descriptions and len(descriptions) != len(labels):
        raise CompileError(
            "AI.CLASSIFY needs one description per label, empty for none"
        )
    for description in descriptions:
        if not isinstance(description, str):
            raise CompileError("an AI.CLASSIFY description is text or empty")
        if len(description.split()) > DESCRIPTION_WORDS:
            raise CompileError(
                f"an AI.CLASSIFY description has at most {DESCRIPTION_WORDS} "
                f"words, got {len(description.split())}"
            )


def validate_task_description(text: str) -> None:
    """Check an AI.CLASSIFY task description's length.

    Raises:
        CompileError: More than TASK_DESCRIPTION_WORDS words.
    """
    if len(text.split()) > TASK_DESCRIPTION_WORDS:
        raise CompileError(
            f"an AI.CLASSIFY task description has at most "
            f"{TASK_DESCRIPTION_WORDS} words, got {len(text.split())}"
        )


def literal_value(value):
    """Return a literal as a plain Python value: a NumPy scalar becomes one."""
    if isinstance(value, tuple):
        return tuple(literal_value(item) for item in value)
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True)
class RegularPredicate:
    """A test of a value expression or a model score against literals.

    ``expression`` reads one source column, with fields or a function
    on it, such as the ``start`` field of an extraction's span; or it
    is an AI.SCORE call, compared with a threshold between 0 and 1.
    ``value`` is the literal compared with, a tuple of literals for
    ``in``, and None for the null tests. A column test is decided
    without the model.
    """

    expression: Expression | ModelCall
    comparison: str
    value: Any = None

    type_name: ClassVar[str] = "quail.regular_predicate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", literal_value(self.value))

    @property
    def column(self) -> ColumnRef:
        """The one source column the expression reads."""
        (column,) = column_references(self.expression)
        return column

    @property
    def function(self) -> Optional[str]:
        """The function applied last, or None."""
        return (self.expression.function
                if isinstance(self.expression, ScalarFunction) else None)

    def aliases(self) -> tuple[str, ...]:
        if isinstance(self.expression, ModelCall):
            return self.expression.aliases()
        return (self.column.alias,)

    @property
    def text(self) -> bool:
        """Whether the comparison tests text against a pattern."""
        return self.comparison.removeprefix("not ") in TEXT_COMPARISONS

    def validate(self) -> None:
        if isinstance(self.expression, ModelCall):
            self.expression.validate()
            if self.expression.kind != "score":
                raise CompileError("only an AI.SCORE call compares with a threshold")
            if self.comparison not in SCORE_COMPARISONS:
                raise CompileError(
                    f"unsupported AI.SCORE comparison {self.comparison!r}")
            if not isinstance(self.value, (int, float)) or not 0 <= self.value <= 1:
                raise CompileError("AI.SCORE threshold must be between 0 and 1")
            return
        if len(column_references(self.expression)) != 1:
            raise CompileError("a column predicate tests one column")
        self.expression.validate()
        if self.comparison not in COLUMN_COMPARISONS:
            raise CompileError(
                f"unsupported column comparison {self.comparison!r}; "
                f"supported: {', '.join(COLUMN_COMPARISONS)}"
            )
        if self.comparison == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise CompileError("IN needs a non-empty list of literals")
            if len(set(self.value)) != len(self.value):
                raise CompileError(f"{self.name} IN lists a value twice")
        elif self.comparison.startswith("is"):
            if self.value is not None:
                raise CompileError("a null test takes no value")
        elif self.text:
            if not isinstance(self.value, str):
                raise CompileError(f"{self.comparison} compares with a string")
        elif self.value is None or isinstance(self.value, (tuple, list)):
            raise CompileError(f"{self.comparison} compares with one literal")

    def validate_text_column(self) -> None:
        """Check the literals of a test over a text column, such as an answer.

        Raises:
            CompileError: The text is compared with a number or boolean,
                or its length with a string.
        """
        self.validate()
        if self.comparison.startswith("is"):
            return
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        if self.function is None and any(not isinstance(v, str) for v in values):
            raise CompileError(
                f"{self.name} is text; {self.comparison} compares it with a string")
        if self.function == "length" and any(
                isinstance(v, (str, bool)) for v in values):
            raise CompileError(f"{self.name} compares with a number")

    def evaluate(self, columns) -> pa.ChunkedArray:
        """Return a boolean array of the rows the predicate accepts.

        A null compared with a literal is neither accepted nor rejected
        and comes back null; accepts() treats it as rejected.

        Args:
            columns: The table, record batch, or rows holding the
                tested column, read with ``column(name)``.

        Raises:
            CompileError: The literal cannot be compared with the
                column's values, such as a string against a number
                column.
        """
        values = evaluate_expression(self.expression, columns)
        try:
            comparison = self.comparison.removeprefix("not ")
            if comparison == "is null":
                mask = pc.is_null(values)
            elif comparison == "is not null":
                mask = pc.is_valid(values)
            elif comparison == "in":
                mask = membership(values, self.value)
            elif isinstance(self.value, str):
                mask = COMPARISON_KERNELS[comparison](values, self.value)
            else:
                mask = compare_literal(values, comparison, self.value)
            return pc.invert(mask) if comparison != self.comparison else mask
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
                pa.ArrowTypeError, TypeError, ValueError, OverflowError) as error:
            raise CompileError(
                f"{self} cannot be evaluated on a column of type "
                f"{values.type}: {error}") from error

    def accepts(self, columns) -> pa.Array:
        """Return the rows the predicate accepts, with every null rejected."""
        return accepted_rows(self.evaluate(columns))

    @property
    def name(self) -> str:
        """The tested expression as written."""
        return _explain(self.expression)

    def __str__(self) -> str:
        if self.comparison == "in":
            return f"{self.name} IN ({', '.join(repr(v) for v in self.value)})"
        if self.comparison.startswith("is"):
            return f"{self.name} {self.comparison.upper()}"
        return f"{self.name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Alias:
    """An expression returned under a name: a model call or a value expression."""

    expression: Expression | ModelCall
    name: str

    type_name: ClassVar[str] = "quail.alias"

    def validate(self) -> None:
        if isinstance(self.expression, ModelCall):
            self.expression.validate()
        else:
            _validate_tree(self.expression)
        if not self.name:
            raise CompileError("a projected expression needs a name")


# a value expression: a tree of these nodes with ColumnRef leaves
Expression = ColumnRef | GetField | ScalarFunction | Alias


def model_call(expression) -> ModelCall:
    """Return the model call inside a predicate or projected expression."""
    if isinstance(expression, ModelCall):
        return expression
    if isinstance(expression, (RegularPredicate, Alias)) and isinstance(
            expression.expression, ModelCall):
        return expression.expression
    raise TypeError(f"{type(expression).__name__} holds no model call")


def is_score(expression) -> bool:
    """Return whether the expression computes an AI.SCORE value."""
    return model_call(expression).kind == "score"


def has_score(plan) -> bool:
    """Return whether a logical plan computes an AI.SCORE value anywhere."""
    operators = plan.operators()
    return (
        any(
            is_score(predicate.expression)
            for predicates in operators.filters.values()
            for predicate in predicates
        )
        or any(is_score(join.predicate) for join in operators.joins)
        or any(
            aliased_call(expression, "score") is not None
            for expression in operators.projections
        )
    )


def validate_predicate(expression) -> None:
    """Check that an expression asks the model yes or no for every row."""
    if isinstance(expression, RegularPredicate) and isinstance(
            expression.expression, ModelCall):
        expression.validate()
        return
    if isinstance(expression, ModelCall):
        expression.validate()
        if expression.kind == "score":
            raise CompileError(
                "an AI.SCORE call is a predicate only when compared with a threshold"
            )
        if expression.kind == "label":
            raise CompileError(
                "an AI.CLASSIFY label is tested by a Filter on its "
                "column, not by a SemanticFilter"
            )
        if expression.kind == "extract":
            raise CompileError(
                "an AI.EXTRACT answer is a result column, not a predicate"
            )
        return
    raise CompileError(
        f"a predicate is a model call or a comparison of an AI.SCORE call, "
        f"got {expression}"
    )


@dataclass(frozen=True)
class FilterPredicate:
    """One predicate over a table's documents, with its selectivity hint."""

    expression: Any  # ModelCall | RegularPredicate over an AI.SCORE call
    selectivity: Optional[float] = None  # fraction of documents that
    #                                       pass; None means not given

    type_name: ClassVar[str] = "quail.filter_predicate"

    @property
    def prompt(self) -> Prompt:
        """The prompt the predicate asks."""
        return model_call(self.expression).prompt


@dataclass(frozen=True)
class Equality:
    """One ordinary join condition: two columns of two tables are equal."""

    left: Any  # a value expression over one table
    right: Any

    type_name: ClassVar[str] = "quail.equality"

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def __str__(self) -> str:
        return f"{column_name(self.left)} = {column_name(self.right)}"


@dataclass(frozen=True)
class ColumnComparison:
    """One ordinary join condition that compares two tables' columns.

    For example, ``r1.id < r2.id`` keeps each unordered pair of a
    self-join once. Equal columns are an Equality instead.
    """

    left: Any  # a value expression over one table
    comparison: str  # one of JOIN_COMPARISONS
    right: Any

    type_name: ClassVar[str] = "quail.column_comparison"

    def __post_init__(self):
        if self.comparison not in JOIN_COMPARISONS:
            raise CompileError(
                f"a join comparison is one of {list(JOIN_COMPARISONS)}, "
                f"not {self.comparison!r}")

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def flipped(self) -> "ColumnComparison":
        """The same condition with its two sides swapped."""
        return ColumnComparison(self.right, FLIPPED_COMPARISONS[self.comparison],
                                self.left)

    def __str__(self) -> str:
        return (f"{column_name(self.left)} {self.comparison} "
                f"{column_name(self.right)}")


JOIN_COMPARISONS = ("<>", "<", "<=", ">", ">=")
FLIPPED_COMPARISONS = {"<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _explain(expression) -> str:
    if isinstance(expression, RegularPredicate):
        return str(expression)
    if isinstance(expression, Alias):
        call = aliased_call(expression)
        if call is None:
            return f"{column_name(expression.expression)} AS {expression.name}"
        if call.kind in ("label", "extract"):
            # the SemanticClassify or SemanticExtract below computes
            # the column
            return expression.name
        return f"{_explain(call)} AS {expression.name}"
    if isinstance(expression, (ColumnRef, GetField, ScalarFunction)):
        return column_name(expression)
    if expression.kind == "label":
        return f"AI.CLASSIFY({expression.prompt.template!r}, {list(expression.labels)})"
    if expression.kind == "extract":
        return f"AI.EXTRACT({expression.prompt.template!r}, {expression.question!r})"
    function = "AI.SCORE" if expression.kind == "score" else "AI.IF"
    return f"{function}({expression.prompt.template!r})"


@dataclass(frozen=True)
class SortKey:
    """One ORDER BY term: a column or a projected expression and its direction."""

    expression: Any  # a value expression, an Alias, or an AggregateCall
    descending: bool = False
    nulls_first: bool = False

    type_name: ClassVar[str] = "quail.sort_key"

    @property
    def name(self) -> str:
        """Return the result column name the key sorts on."""
        return column_name(self.expression)

    def validate(self) -> None:
        if isinstance(self.expression, AggregateCall):
            self.expression.validate()
        elif not isinstance(self.expression, WindowCall):
            _validate_tree(self.expression)

    def __str__(self) -> str:
        return (
            f"{self.name} {'DESC' if self.descending else 'ASC'} "
            f"NULLS {'FIRST' if self.nulls_first else 'LAST'}"
        )


AGGREGATE_FUNCTIONS = ("count", "count_distinct", "sum", "avg", "min", "max")

WINDOW_FUNCTIONS = ("row_number", "lag", "lead", "sum", "count")


@dataclass(frozen=True)
class WindowCall:
    """One function over the rows of each row's partition, under a name.

    ``argument`` names a projected column, or is None for row_number
    and count(*). ``partition`` names the projected columns that group
    the rows, and ``order`` the (column, descending, nulls_first) keys
    that order each partition. sum and count run from the partition's
    first row to the current one.
    """

    function: str
    argument: Optional[str]
    partition: tuple[str, ...]
    order: tuple[tuple[str, bool, bool], ...]
    name: str

    type_name: ClassVar[str] = "quail.window_call"

    def validate(self, projected: tuple) -> None:
        """Check the function and that every column it reads is projected."""
        if self.function not in WINDOW_FUNCTIONS:
            raise CompileError(
                f"unsupported window function {self.function!r}; supported: "
                f"{', '.join(WINDOW_FUNCTIONS)}")
        if self.function == "row_number" and self.argument is not None:
            raise CompileError("row_number() takes no column")
        if self.function in ("lag", "lead", "sum") and self.argument is None:
            raise CompileError(f"{self.function}() needs a column")
        if self.function in ("lag", "lead") and not self.order:
            raise CompileError(f"{self.function}() needs an ORDER BY in its window")
        for column in (self.argument, *self.partition,
                       *(name for name, _, _ in self.order)):
            if column is not None and column not in projected:
                raise CompileError(
                    f"the window {self.name!r} reads {column!r}, which is not "
                    f"a projected column")

    def __str__(self) -> str:
        clauses = []
        if self.partition:
            clauses.append("partition by " + ", ".join(self.partition))
        if self.order:
            clauses.append("order by " + ", ".join(
                f"{name}{' desc' if descending else ''}"
                for name, descending, _ in self.order))
        return (f"{self.name} = {self.function}({self.argument or ''}) "
                f"over ({' '.join(clauses)})")


HAVING_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class AggregateCall:
    """One aggregate over the rows of a group, returned under a name.

    ``argument`` names a projected column, or is None for ``count(*)``.
    """

    function: str
    argument: Optional[str]
    name: str

    type_name: ClassVar[str] = "quail.aggregate_call"

    def validate(self) -> None:
        if self.function not in AGGREGATE_FUNCTIONS:
            raise CompileError(
                f"unsupported aggregate {self.function!r}; supported: "
                f"{', '.join(AGGREGATE_FUNCTIONS)}"
            )
        if self.argument is None and self.function != "count":
            raise CompileError(f"{self.function}(*) is not an aggregate")
        if not self.name:
            raise CompileError("an aggregate needs a name")

    def __str__(self) -> str:
        function = (
            "count(distinct "
            if self.function == "count_distinct"
            else f"{self.function}("
        )
        return f"{self.name} = {function}{self.argument or '*'})"


@dataclass(frozen=True)
class HavingTest:
    """A comparison of one aggregate with a literal, keeping the group."""

    aggregate: AggregateCall
    comparison: str
    value: Any

    type_name: ClassVar[str] = "quail.having_test"

    def validate(self) -> None:
        self.aggregate.validate()
        if self.comparison not in HAVING_COMPARISONS:
            raise CompileError(
                f"HAVING compares with one of {HAVING_COMPARISONS}, got "
                f"{self.comparison!r}"
            )
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise CompileError("HAVING compares an aggregate with a number")

    def __post_init__(self) -> None:
        value = self.value
        object.__setattr__(self, "value", value.item() if hasattr(value, "item")
                           and not isinstance(value, (int, float)) else value)

    def __str__(self) -> str:
        return f"{self.aggregate.name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Aggregation:
    """GROUP BY keys, the aggregates over each group, and the HAVING tests.

    ``keys`` and each aggregate's argument name projected columns.
    ``output`` lists the result columns in SELECT order: keys and
    aggregate names. An aggregate only a HAVING test reads is absent
    from ``output``.
    """

    keys: tuple[str, ...]
    aggregates: tuple[AggregateCall, ...]
    output: tuple[str, ...]
    having: tuple[HavingTest, ...] = ()

    type_name: ClassVar[str] = "quail.aggregation"

    def validate(self, projected: tuple) -> None:
        """Check the keys, arguments, output, and tests against the projection.

        Args:
            projected: The names of the projected columns.
        """
        for key in self.keys:
            if key not in projected:
                raise CompileError(f"GROUP BY {key!r} is not a projected column")
        names = list(self.keys)
        for aggregate in self.aggregates:
            aggregate.validate()
            if aggregate.argument is not None and aggregate.argument not in projected:
                raise CompileError(
                    f"{aggregate} reads a column the projection does not have"
                )
            names.append(aggregate.name)
        if len(names) != len(set(names)):
            raise CompileError(
                f"GROUP BY keys and aggregate names must be unique, got {names}"
            )
        if not self.output:
            raise CompileError("an aggregation returns at least one column")
        for name in self.output:
            if name not in names:
                raise CompileError(
                    f"{name!r} is neither a GROUP BY key nor an aggregate"
                )
        if len(self.output) != len(set(self.output)):
            raise CompileError(f"output names must be unique, got {self.output}")
        for test in self.having:
            if not isinstance(test, HavingTest):
                raise CompileError(
                    f"a HAVING term is a HavingTest, got {type(test).__name__}"
                )
            test.validate()
            if test.aggregate not in self.aggregates:
                raise CompileError(
                    f"HAVING {test} tests an aggregate the aggregation does not compute"
                )

    def __str__(self) -> str:
        parts = []
        if self.keys:
            parts.append("group by " + ", ".join(self.keys))
        parts.append(", ".join(str(a) for a in self.aggregates))
        if self.having:
            parts.append("having " + " and ".join(str(t) for t in self.having))
        return "; ".join(parts)
