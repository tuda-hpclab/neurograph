#pragma once

/*
 * This file is part of the neurograph software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"

#include "graph/DistributedGraph.h"
#include "metrics/paths/DeltaSteppingVariants/PullModelVariant.h"
#include "metrics/paths/DeltaSteppingVariants/PushBruteForceVariant.h"
#include "metrics/paths/DeltaSteppingVariants/PushCheckVariant.h"
#include "metrics/paths/DeltaSteppingVariants/PushInnerShortVariant.h"
#include "metrics/paths/DeltaSteppingVariants/PushShortLongVariant.h"
#include "metrics/paths/types/SsspResult.h"

#include <cpp-utility/Exception.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/reductions/MPIReductions.h>
#include <mpi-wrapper/rma/RMAWindow.h>

#include <limits>
#include <span>

class DeltaStepping {
    constexpr static auto max_distance = std::numeric_limits<distance_type>::max();

public:
    [[nodiscard]] static SsspResult<distance_type> compute_shortest_distances(const DistributedGraph& graph, const NodeIdentifier root_node, const distance_type delta,
                                                                               const DeltaSteppingEpochType epoch_type) {
        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto my_rank_cast = my_rank.get_rank();
        const auto number_local_nodes = graph.get_number_local_nodes();

        auto distance_window = mpiPP::RMAWindow<distance_type>(number_local_nodes);

        for (auto i = 0U; i < number_local_nodes; i++) {
            distance_window.put(max_distance, i, my_rank);
        }

        if (root_node.owning_mpi_rank == my_rank_cast) {
            distance_window.put(0, root_node.node_id, my_rank);
        }

        switch (epoch_type) {
        case DeltaSteppingEpochType::PushBruteForce:
            PushBruteForceVariant::compute_epochs(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushCheck:
            PushCheckVariant::compute_epochs(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushShortLong:
            PushShortLongVariant::compute_epochs(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PushInnerShort:
            PushInnerShortVariant::compute_epochs(graph, root_node, delta, distance_window);
            break;
        case DeltaSteppingEpochType::PullModel:
            PullModelVariant::compute_epochs(graph, root_node, delta, distance_window);
            break;
        default:
            utility::Exception::fail("Epoch type is unknown");
        }

        const auto span = std::span{ distance_window.get_pointer(), distance_window.get_local_size() };

        const auto [local_sum_distances, local_max_distance, local_sum_efficiency, local_number_unreached] = summarize_distances<distance_type>(span);

        const auto global_sum_distances = mpiPP::MPIReductions::reduce_sum(local_sum_distances);
        const auto global_max_distance = mpiPP::MPIReductions::reduce_max(local_max_distance);
        const auto global_sum_efficiency = mpiPP::MPIReductions::reduce_sum(local_sum_efficiency);
        const auto global_number_unreached = mpiPP::MPIReductions::reduce_sum(local_number_unreached);

        return { global_sum_distances, global_max_distance, global_sum_efficiency, global_number_unreached };
    }
};
