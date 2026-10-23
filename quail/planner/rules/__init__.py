"""Built in logical and physical rules in execution order."""

from quail.planner.rules.logical import (
    ShouldEliminateDistinct,
    ShouldLiftMaps,
    ShouldPushDownDistinct,
    ShouldPushDownFilters,
    ShouldPushDownProjection,
)
from quail.planner.rules.physical import (
    ChooseAttentionAlgorithm,
    ChooseClassifyImplementation,
    ShouldAskOncePerDocument,
    ShouldEliminateWindowSort,
    ShouldPushDownLimit,
    ShouldPushDownLimitThroughWindow,
    ShouldRetainKv,
    ShouldSharePrefixes,
)


def built_in_rules() -> tuple:
    """Return logical rules followed by physical rules in dependency order.

    Classification implementation determines pipeline compatibility for
    KV retention. Prefix sharing prices the writes retention requires.
    Tree attention uses the selected shared prefixes.
    """
    return (
        ShouldEliminateDistinct(), ShouldPushDownDistinct(), ShouldPushDownProjection(),
        ShouldPushDownFilters(), ShouldLiftMaps(), ShouldEliminateWindowSort(),
        ShouldPushDownLimitThroughWindow(), ShouldPushDownLimit(),
        ChooseClassifyImplementation(), ShouldRetainKv(), ShouldSharePrefixes(),
        ChooseAttentionAlgorithm(), ShouldAskOncePerDocument(),
    )
