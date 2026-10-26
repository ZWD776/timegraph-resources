from .json_to_graph import JSONToGraphTransformer


from .temporal_graph_to_canonical_graph import (
    TemporalGraphToCanonicalGraphConverter,
    build_canonical_temporal_memory_graph,
)
from .canonical_graph_time_axis_compaction import (
    TimeAxisCompactionResult,
    compact_canonical_graph_by_time_axis,
)
from .canonical_graph_structural_edge_virtualization import (
    DEFAULT_VIRTUALIZED_STRUCTURAL_EDGE_TYPES,
    StructuralEdgeVirtualizationResult,
    virtualize_canonical_graph_structure_edges,
)
from .canonical_graph_stable_core_compaction import (
    StableCoreCompactionResult,
    compact_canonical_graph_to_stable_core,
)
from .canonical_graph_entity_segment_scene_compaction import (
    EntitySegmentSceneCompactionResult,
    compact_entity_segments_by_scene_span,
)
from .canonical_graph_relation_scene_span_compaction import (
    RelationSceneSpanCompactionResult,
    compact_relations_by_scene_span,
)
from .canonical_graph_scene_fact_span_compaction import (
    SceneFactSpanCompactionResult,
    compact_scene_facts_by_span,
)
from .canonical_graph_topology_pruning import (
    DEFAULT_PRUNED_EPISODE_KINDS,
    TopologyPruningResult,
    prune_low_information_topology,
)
from .time_fact_core_pipeline import (
    SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
    SECOND_MAINLINE_CONCEPTUAL_NAME,
    SECOND_MAINLINE_DISPLAY_NAME,
    SECOND_MAINLINE_PROFILE_ID,
    TIME_FACT_CORE_DERIVATION_STEPS,
    TIME_FACT_CORE_DESIGN_PRINCIPLES,
    TimeFactCoreGraphResult,
    TopologyPrunedAfterSceneFactResult,
    build_time_fact_core_graph,
    build_topology_pruned_after_scene_fact_graph,
)
from .temporal_graph_to_memory_ir import (
    TemporalGraphToMemoryIRConverter,
    build_memory_ir,
)
from .temporal_graph_to_stable_identity import (
    TemporalGraphToStableIdentityConverter,
    build_stable_identity_document,
)

__all__ = [
    "JSONToGraphTransformer",
    "TemporalGraphToCanonicalGraphConverter",
    "TemporalGraphToMemoryIRConverter",
    "TemporalGraphToStableIdentityConverter",
    "TimeAxisCompactionResult",
    "DEFAULT_VIRTUALIZED_STRUCTURAL_EDGE_TYPES",
    "StructuralEdgeVirtualizationResult",
    "StableCoreCompactionResult",
    "EntitySegmentSceneCompactionResult",
    "RelationSceneSpanCompactionResult",
    "SceneFactSpanCompactionResult",
    "DEFAULT_PRUNED_EPISODE_KINDS",
    "SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID",
    "SECOND_MAINLINE_CONCEPTUAL_NAME",
    "SECOND_MAINLINE_DISPLAY_NAME",
    "SECOND_MAINLINE_PROFILE_ID",
    "TIME_FACT_CORE_DERIVATION_STEPS",
    "TIME_FACT_CORE_DESIGN_PRINCIPLES",
    "TimeFactCoreGraphResult",
    "TopologyPruningResult",
    "TopologyPrunedAfterSceneFactResult",
    "build_canonical_temporal_memory_graph",
    "build_time_fact_core_graph",
    "build_topology_pruned_after_scene_fact_graph",
    "compact_canonical_graph_by_time_axis",
    "compact_canonical_graph_to_stable_core",
    "compact_entity_segments_by_scene_span",
    "compact_relations_by_scene_span",
    "compact_scene_facts_by_span",
    "prune_low_information_topology",
    "virtualize_canonical_graph_structure_edges",
    "build_memory_ir",
    "build_stable_identity_document",
]
