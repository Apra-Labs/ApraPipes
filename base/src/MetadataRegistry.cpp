#include "MetadataRegistry.h"
#include "Logger.h"
#include <queue>
#include <set>
#include <limits>

MetadataRegistry& MetadataRegistry::getInstance() {
    static MetadataRegistry instance;
    static bool initialized = false;

    if (!initialized) {
        // Auto-initialize built-in conversions on first access
        // This ensures conversions are available during module validation
        registerBuiltinConversions();
        initialized = true;
    }

    return instance;
}

void MetadataRegistry::registerConversion(
    FrameMetadata::FrameType source,
    FrameMetadata::FrameType target,
    ConverterFunc converter,
    int cost)
{
    if (!converter) {
        LOG_ERROR << "Cannot register null converter from " << source << " to " << target;
        return;
    }

    if (cost <= 0) {
        LOG_ERROR << "Conversion cost must be positive, got " << cost;
        return;
    }

    ConversionStep step(source, target, converter, cost);
    mConversions[source].push_back(step);

    LOG_INFO << "Registered conversion: " << source << " -> " << target
             << " (cost: " << cost << ")";
}

std::optional<ConversionPath> MetadataRegistry::findConversionPath(
    FrameMetadata::FrameType source,
    FrameMetadata::FrameType target) const
{
    // Direct match - no conversion needed
    if (source == target) {
        return ConversionPath();  // Empty path, zero cost
    }

    // Dijkstra's algorithm for shortest path
    // Priority queue: (cost, currentType, path)
    using QueueItem = std::tuple<int, FrameMetadata::FrameType, ConversionPath>;
    auto cmp = [](const QueueItem& a, const QueueItem& b) {
        return std::get<0>(a) > std::get<0>(b);  // Min-heap by cost
    };
    std::priority_queue<QueueItem, std::vector<QueueItem>, decltype(cmp)> pq(cmp);

    // Track best cost to reach each type
    std::map<FrameMetadata::FrameType, int> bestCost;

    // Start search from source
    ConversionPath initialPath;
    pq.push({0, source, initialPath});
    bestCost[source] = 0;

    while (!pq.empty()) {
        auto [currentCost, currentType, currentPath] = pq.top();
        pq.pop();

        // Found target
        if (currentType == target) {
            LOG_INFO << "Found conversion path from " << source << " to " << target
                     << " (cost: " << currentCost << ", steps: " << currentPath.steps.size() << ")";
            return currentPath;
        }

        // Skip if we've already found a better path to this type
        if (bestCost.count(currentType) && bestCost[currentType] < currentCost) {
            continue;
        }

        // Explore all conversions from current type
        auto it = mConversions.find(currentType);
        if (it == mConversions.end()) {
            continue;
        }

        for (const auto& step : it->second) {
            int newCost = currentCost + step.cost;

            // Skip if we've already found a better path to this type
            if (bestCost.count(step.targetType) && bestCost[step.targetType] <= newCost) {
                continue;
            }

            // Create new path with this step
            ConversionPath newPath = currentPath;
            newPath.addStep(step);

            bestCost[step.targetType] = newCost;
            pq.push({newCost, step.targetType, newPath});
        }
    }

    // No path found
    LOG_WARNING << "No conversion path found from " << source << " to " << target;
    return std::nullopt;
}

frame_sp MetadataRegistry::convertFrame(
    const frame_sp& sourceFrame,
    FrameMetadata::FrameType targetType) const
{
    if (!sourceFrame) {
        LOG_ERROR << "Cannot convert null frame";
        return nullptr;
    }

    FrameMetadata::FrameType sourceType = sourceFrame->getMetadata()->getFrameType();

    // No conversion needed
    if (sourceType == targetType) {
        return sourceFrame;
    }

    // Find conversion path
    auto pathOpt = findConversionPath(sourceType, targetType);
    if (!pathOpt.has_value()) {
        LOG_ERROR << "Cannot convert frame: no conversion path from "
                  << sourceType << " to " << targetType;
        return nullptr;
    }

    ConversionPath path = pathOpt.value();

    // Direct match (empty path)
    if (path.isEmpty()) {
        return sourceFrame;
    }

    // Apply conversion steps sequentially
    frame_sp currentFrame = sourceFrame;
    for (size_t i = 0; i < path.steps.size(); ++i) {
        const auto& step = path.steps[i];

        LOG_TRACE << "Applying conversion step " << (i + 1) << "/" << path.steps.size()
                  << ": " << step.sourceType << " -> " << step.targetType;

        currentFrame = step.converter(currentFrame);

        if (!currentFrame) {
            LOG_ERROR << "Conversion failed at step " << (i + 1)
                      << " (" << step.sourceType << " -> " << step.targetType << ")";
            return nullptr;
        }
    }

    LOG_DEBUG << "Successfully converted frame from " << sourceType
              << " to " << targetType << " in " << path.steps.size() << " steps";

    return currentFrame;
}

bool MetadataRegistry::areCompatible(
    FrameMetadata::FrameType source,
    FrameMetadata::FrameType target) const
{
    // Direct match
    if (source == target) {
        return true;
    }

    // Check if conversion path exists
    auto pathOpt = findConversionPath(source, target);
    return pathOpt.has_value();
}

std::vector<FrameMetadata::FrameType> MetadataRegistry::getCompatibleOutputTypes(
    FrameMetadata::FrameType source) const
{
    std::vector<FrameMetadata::FrameType> compatibleTypes;

    // Add the source type itself
    compatibleTypes.push_back(source);

    // BFS to find all reachable types
    std::queue<FrameMetadata::FrameType> queue;
    std::set<FrameMetadata::FrameType> visited;

    queue.push(source);
    visited.insert(source);

    while (!queue.empty()) {
        FrameMetadata::FrameType current = queue.front();
        queue.pop();

        auto it = mConversions.find(current);
        if (it == mConversions.end()) {
            continue;
        }

        for (const auto& step : it->second) {
            if (visited.find(step.targetType) == visited.end()) {
                visited.insert(step.targetType);
                compatibleTypes.push_back(step.targetType);
                queue.push(step.targetType);
            }
        }
    }

    return compatibleTypes;
}

std::map<FrameMetadata::FrameType, std::vector<FrameMetadata::FrameType>>
MetadataRegistry::getRegisteredConversions() const
{
    std::map<FrameMetadata::FrameType, std::vector<FrameMetadata::FrameType>> result;

    for (const auto& [source, steps] : mConversions) {
        std::vector<FrameMetadata::FrameType> targets;
        for (const auto& step : steps) {
            targets.push_back(step.targetType);
        }
        result[source] = targets;
    }

    return result;
}

void MetadataRegistry::clear() {
    mConversions.clear();
    LOG_INFO << "MetadataRegistry cleared";
}
