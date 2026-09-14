/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "StronglyConnectedComponents.h"

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/connectivity/SccVariants/ColoringVariant.h"
#include "metrics/connectivity/SccVariants/DistributedSccState.h"
#include "metrics/connectivity/SccVariants/ForwardBackwardVariant.h"
#include "metrics/connectivity/SccVariants/MultistepVariant.h"
#include "metrics/connectivity/SccVariants/TarjanGatheredVariant.h"

#include <cpp-utility/Exception.hpp>
#include <cpp-utility/profiling/RegionTimer.hpp>

#include <mpi-wrapper/instrumentation/MPIRegionTimerReport.h>

SccResult StronglyConnectedComponents::compute_strongly_connected_components(const DistributedGraph& graph, const SccAlgorithm algorithm,
                                                                            const global_node_id_type tail_threshold) {
    auto timer_setup = utility::RegionTimer{ "scc-setup" };

    // Collects the node distribution and the degrees, so all ranks have to take part
    auto state = DistributedSccState{ graph };

    timer_setup.stop();

    auto timer_compute = utility::RegionTimer{ "scc-compute" };

    const auto result = [&state, algorithm, tail_threshold]() -> SccResult {
        switch (algorithm) {
        case SccAlgorithm::TarjanGathered:
            return TarjanGatheredVariant::all_compute_components(state);
        case SccAlgorithm::ForwardBackward:
            return ForwardBackwardVariant::all_compute_components(state);
        case SccAlgorithm::Coloring:
            return ColoringVariant::all_compute_components(state);
        case SccAlgorithm::Multistep:
            return MultistepVariant::all_compute_components(state, tail_threshold);
        default:
            utility::Exception::fail("StronglyConnectedComponents::compute_strongly_connected_components: the algorithm is unknown");
        }
    }();

    timer_compute.stop();

    mpiPP::MPIRegionTimerReport::reduce_and_print("StronglyConnectedComponents");
    utility::RegionTimer::init();

    return result;
}
