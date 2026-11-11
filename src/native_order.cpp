// Same DFS and earliest-minimum cycle cut as src/durner/native.py.
#include <cstdint>
#include <unordered_set>
#include <vector>

extern "C" int native_order(
    int n, int nactive, const int* active, const int* offsets,
    const int* successors, const double* departure, const double* arrival,
    int* order, int* cut_sources, int* cut_targets, std::uint64_t* work) {
    std::vector<unsigned char> visited(n, 0);
    std::vector<int> position(n, -1), stack, indices;
    std::unordered_set<std::uint64_t> cuts;
    int ordered = 0, ncuts = 0;
    std::uint64_t visits = 0, cycle_edges = 0;
    auto edge_key = [](int a, int b) {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
    };
    for (int a = 0; a < nactive; ++a) {
        const int anchor = active[a];
        if (visited[anchor] == 2) continue;
        stack.assign(1, anchor);
        indices.assign(1, offsets[anchor + 1] - offsets[anchor]);
        position[anchor] = 0;
        while (!stack.empty()) {
            ++visits;
            const int current = stack.back();
            visited[current] = 1;
            if (indices.back() == 0) {
                order[ordered++] = current;
                visited[current] = 2;
                position[current] = -1;
                stack.pop_back();
                indices.pop_back();
                continue;
            }
            const int nxt = successors[offsets[current] + --indices.back()];
            if (cuts.count(edge_key(current, nxt))) continue;
            if (visited[nxt] == 1) {
                const int start = position[nxt], end = static_cast<int>(stack.size()) - 1;
                int selected = start;
                int first = stack[start], second = start < end ? stack[start + 1] : nxt;
                double minimum = departure[second] - arrival[first];
                for (int i = start + 1; i <= end; ++i) {
                    const int b = i < end ? stack[i + 1] : nxt;
                    const double value = departure[b] - arrival[stack[i]];
                    if (value < minimum) {
                        minimum = value;
                        selected = i;
                    }
                }
                cycle_edges += end - start + 1;
                first = stack[selected];
                second = selected < end ? stack[selected + 1] : nxt;
                cuts.insert(edge_key(first, second));
                cut_sources[ncuts] = first;
                cut_targets[ncuts++] = second;
                if (second != nxt) {
                    while (stack.back() != first) {
                        visited[stack.back()] = 0;
                        position[stack.back()] = -1;
                        stack.pop_back();
                        indices.pop_back();
                    }
                }
            } else if (visited[nxt] != 2) {
                position[nxt] = static_cast<int>(stack.size());
                stack.push_back(nxt);
                indices.push_back(offsets[nxt + 1] - offsets[nxt]);
            }
        }
    }
    work[0] = visits;
    work[1] = cycle_edges;
    return ncuts;
}
