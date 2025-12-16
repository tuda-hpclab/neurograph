FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        4160d259d961cd393fd8d67590a8c7d210207348
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
	
	list(APPEND visual_studio_additional_files "metrics/AllPairsDistances.h")
	list(APPEND visual_studio_additional_files "metrics/AllPairsDistancesHistogram.h")
	list(APPEND visual_studio_additional_files "metrics/AllPairsShortestPath.h")
	list(APPEND visual_studio_additional_files "metrics/ArcCounter.h")
	list(APPEND visual_studio_additional_files "metrics/ArcLength.h")
	list(APPEND visual_studio_additional_files "metrics/ArcLengthHistogram.h")
	list(APPEND visual_studio_additional_files "metrics/AreaConnectivity.h")
	list(APPEND visual_studio_additional_files "metrics/Assortativity.h")
	list(APPEND visual_studio_additional_files "metrics/Centrality.h")
	list(APPEND visual_studio_additional_files "metrics/Clustering.h")
	list(APPEND visual_studio_additional_files "metrics/DeltaStepping.h")
	list(APPEND visual_studio_additional_files "metrics/Degree.h")
	list(APPEND visual_studio_additional_files "metrics/DegreeCounter.h")
	list(APPEND visual_studio_additional_files "metrics/DegreeHistogram.h")
	list(APPEND visual_studio_additional_files "metrics/DiameterApproximation.h")
	list(APPEND visual_studio_additional_files "metrics/Dijkstra.h")
	list(APPEND visual_studio_additional_files "metrics/Distance.h")
	list(APPEND visual_studio_additional_files "metrics/NetworkMotifs.h")
	list(APPEND visual_studio_additional_files "metrics/NodeCounter.h")
	list(APPEND visual_studio_additional_files "metrics/WeightCounter.h")

	list(APPEND visual_studio_additional_files "utility/DataCache.h")
	list(APPEND visual_studio_additional_files "utility/Histogram.h")
	list(APPEND visual_studio_additional_files "utility/MemoryPool.h")
	list(APPEND visual_studio_additional_files "utility/MetricStore.h")
	list(APPEND visual_studio_additional_files "utility/PoolAllocator.h")
	list(APPEND visual_studio_additional_files "utility/Status.h")
	list(APPEND visual_studio_additional_files "utility/Timer.h")
	list(APPEND visual_studio_additional_files "utility/Vec3.h")
endif()

add_library(graph_lib STATIC)
target_sources(
  graph_lib
  PRIVATE
  "graph/ArcTransformer.cpp"
  "graph/DistributedGraph.cpp"
  "graph/FileLoader.cpp"
  "metrics/AreaConnectivity.cpp"
  "metrics/NetworkMotifs.cpp"
  "metrics/Assortativity.cpp"
  "utility/MetricStore.cpp"
  ${visual_studio_additional_files})

target_include_directories(graph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(graph_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(graph_lib PUBLIC project_options project_libraries)

get_target_property(graph_lib_source graph_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${graph_lib_source})

add_executable(ScalableGraphAlgorithm)
target_sources(ScalableGraphAlgorithm PRIVATE "ScalableGraphAlgorithm.cpp")
target_link_libraries(ScalableGraphAlgorithm PRIVATE graph_lib)

target_link_libraries(ScalableGraphAlgorithm PUBLIC project_options project_libraries)
target_include_directories(ScalableGraphAlgorithm SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)
target_include_directories(ScalableGraphAlgorithm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ScalableGraphAlgorithm PRIVATE ${MPI_LIBRARIES})

get_target_property(ScalableGraphAlgorithm_source ScalableGraphAlgorithm SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${ScalableGraphAlgorithm_source})
