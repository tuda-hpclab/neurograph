FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(CLI11)
get_target_property(cli11_includes CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                       "${cli11_includes}")

set(visual_studio_additional_files "" CACHE INTERNAL "")

if(WIN32)
	list(APPEND visual_studio_additional_files "Types.h")

	list(APPEND visual_studio_additional_files "graph/Arc.h")
	list(APPEND visual_studio_additional_files "graph/ArcTransformer.h")
	list(APPEND visual_studio_additional_files "graph/DistributedGraph.h")
	list(APPEND visual_studio_additional_files "graph/FileLoader.h")
	list(APPEND visual_studio_additional_files "graph/GraphTypes.h")
	
	list(APPEND visual_studio_additional_files "metrics/community/AreaConnectivity.h")
	list(APPEND visual_studio_additional_files "metrics/community/Modularity.h")

	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/CommunityDetectionCommon.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/CommunityGraph.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/FlowNetwork.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/InfomapVariant.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/LabelPropagationVariant.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/LeidenVariant.h")
	list(APPEND visual_studio_additional_files "metrics/community/ModularityVariants/LouvainVariant.h")

	list(APPEND visual_studio_additional_files "metrics/connectivity/StronglyConnectedComponents.h")

	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/ColoringVariant.h")
	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/DistributedSccState.h")
	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/ForwardBackwardVariant.h")
	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/MultistepVariant.h")
	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/SccCommon.h")
	list(APPEND visual_studio_additional_files "metrics/connectivity/SccVariants/TarjanGatheredVariant.h")

	list(APPEND visual_studio_additional_files "metrics/counting/InArcCounter.h")
	list(APPEND visual_studio_additional_files "metrics/counting/InWeightCounter.h")
	list(APPEND visual_studio_additional_files "metrics/counting/NodeCounter.h")
	list(APPEND visual_studio_additional_files "metrics/counting/NodeDistributionCounter.h")
	list(APPEND visual_studio_additional_files "metrics/counting/OutArcCounter.h")
	list(APPEND visual_studio_additional_files "metrics/counting/OutWeightCounter.h")

	list(APPEND visual_studio_additional_files "metrics/degree/AverageDegree.h")
	list(APPEND visual_studio_additional_files "metrics/degree/InDegree.h")
	list(APPEND visual_studio_additional_files "metrics/degree/InDegreeHistogram.h")
	list(APPEND visual_studio_additional_files "metrics/degree/OutDegree.h")
	list(APPEND visual_studio_additional_files "metrics/degree/OutDegreeHistogram.h")

	list(APPEND visual_studio_additional_files "metrics/flow/MaximumFlow.h")

	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/DinicVariant.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/DistributedPushRelabelVariant.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/EdmondsKarpVariant.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/FlowNetwork.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/FordFulkersonVariant.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/MaxFlowCommon.h")
	list(APPEND visual_studio_additional_files "metrics/flow/MaxFlowVariants/OrlinKingRaoTarjanVariant.h")

	list(APPEND visual_studio_additional_files "metrics/geometry/AllPairsDistances.h")
	list(APPEND visual_studio_additional_files "metrics/geometry/AllPairsDistancesHistogram.h")
	list(APPEND visual_studio_additional_files "metrics/geometry/ArcLength.h")
	list(APPEND visual_studio_additional_files "metrics/geometry/ArcLengthHistogram.h")

	list(APPEND visual_studio_additional_files "metrics/local_structure/Assortativity.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/Clustering.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifs.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/Reciprocity.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/RichClub.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/Transitivity.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/TriangleCounting.h")

	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/MotifArcCache.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/MotifClassification.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaAccumulateVariant.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaCommon.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaFairVariant.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaFewerArcsVariant.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaFewerMessagesVariant.h")
	list(APPEND visual_studio_additional_files "metrics/local_structure/NetworkMotifsVariants/RmaVariant.h")

	list(APPEND visual_studio_additional_files "metrics/paths/AllPairsShortestPath.h")
	list(APPEND visual_studio_additional_files "metrics/paths/BetweennessCentrality.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaStepping.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DiameterApproximation.h")
	list(APPEND visual_studio_additional_files "metrics/paths/Dijkstra.h")

	list(APPEND visual_studio_additional_files "metrics/paths/types/NodePath.h")
	list(APPEND visual_studio_additional_files "metrics/paths/types/SsspResult.h")
	list(APPEND visual_studio_additional_files "metrics/paths/types/VertexDistance.h")

	list(APPEND visual_studio_additional_files "metrics/paths/BetweennessCentralityVariants/BrandesVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/BetweennessCentralityVariants/PathEnumerationVariant.h")

	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/Buckets.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/EpochCommon.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/PullModelVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/PushBruteForceVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/PushCheckVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/PushInnerShortVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DeltaSteppingVariants/PushShortLongVariant.h")

	list(APPEND visual_studio_additional_files "metrics/paths/DijkstraVariants/BucketQueueVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DijkstraVariants/DijkstraCommon.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DijkstraVariants/FibonacciHeapVariant.h")
	list(APPEND visual_studio_additional_files "metrics/paths/DijkstraVariants/PriorityQueueVariant.h")

	list(APPEND visual_studio_additional_files "utility/MetricStore.h")
	list(APPEND visual_studio_additional_files "utility/RandomNumberGenerator.h")
	list(APPEND visual_studio_additional_files "utility/Vec3.h")
endif()

add_library(graph_lib STATIC)
target_sources(
  graph_lib
  PRIVATE
  "graph/ArcTransformer.cpp"
  "graph/DistributedGraph.cpp"
  "graph/FileLoader.cpp"
  "metrics/community/AreaConnectivity.cpp"
  "metrics/community/Modularity.cpp"
  "metrics/connectivity/StronglyConnectedComponents.cpp"
  "metrics/flow/MaxFlowVariants/DistributedPushRelabelVariant.cpp"
  "metrics/flow/MaximumFlow.cpp"
  "metrics/local_structure/NetworkMotifs.cpp"
  "metrics/local_structure/Assortativity.cpp"
  "metrics/local_structure/Reciprocity.cpp"
  "metrics/local_structure/RichClub.cpp"
  "metrics/local_structure/Transitivity.cpp"
  "utility/MetricStore.cpp"
  ${visual_studio_additional_files})

target_include_directories(graph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(graph_lib PUBLIC project_options project_libraries)

get_target_property(graph_lib_source graph_lib SOURCES)
# PROJECT_SOURCE_DIR, not CMAKE_SOURCE_DIR: the project can also be added with
# add_subdirectory, and the tree root of a source group has to be a prefix of the
# files it groups.
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${graph_lib_source})

add_executable(neurograph)
target_sources(neurograph PRIVATE "neurograph.cpp")
# graph_lib links project_options/project_libraries (incl. MPI::MPI_CXX) and the
# source include dir as PUBLIC, so all of that arrives transitively here.
target_link_libraries(neurograph PRIVATE graph_lib)
# CLI11 is only used by the executable
target_include_directories(neurograph SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)

get_target_property(neurograph_source neurograph SOURCES)
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${neurograph_source})
