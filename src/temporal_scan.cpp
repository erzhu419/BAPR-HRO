#include <algorithm>
#include <vector>

// State indices distinguish a ready passenger from an onboard (trip, stop)
// passenger. The latter may continue on a no-pickup connection of that trip.
extern "C" int temporal_arrival(
    int n, int state_count, const int* arcs, const int* departures,
    const int* arrivals, const unsigned char* canceled, const unsigned char* unavailable,
    int origin, int onboard, int goal, int start, int origin_ready, int deadline,
    const int* transfers) {
    if (origin == goal) return start;
    std::vector<int> order;
    order.reserve(n);
    for (int cid = 0; cid < n; ++cid) {
        if (!canceled[cid] && !unavailable[cid] && departures[cid] >= start &&
            arrivals[cid] >= departures[cid] && arrivals[cid] <= deadline) {
            order.push_back(cid);
        }
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return departures[a] < departures[b];
    });
    std::vector<int> ready(state_count, deadline + 1), head(state_count, -1);
    std::vector<int> next(2 * n), touched, queue;
    std::vector<unsigned char> queued(n, 0);
    ready[origin] = origin_ready;
    if (onboard != -1) ready[onboard] = start;
    int best = deadline;

    auto enqueue = [&](int cid) {
        if (!queued[cid]) {
            queued[cid] = 1;
            queue.push_back(cid);
        }
    };
    for (std::size_t left = 0; left < order.size();) {
        const int time = departures[order[left]];
        if (time >= best) break;
        std::size_t right = left;
        touched.clear();
        queue.clear();
        while (right < order.size() && departures[order[right]] == time) {
            const int cid = order[right++];
            for (int channel = 0; channel < 2; ++channel) {
                const int state = arcs[4 * cid + channel];
                if (state < 0) continue;
                if (ready[state] <= time) {
                    enqueue(cid);
                } else {
                    if (head[state] == -1) touched.push_back(state);
                    const int entry = 2 * cid + channel;
                    next[entry] = head[state];
                    head[state] = entry;
                }
            }
        }

        auto relax = [&](int state, int new_time) {
            if (state < 0 || new_time >= ready[state]) return;
            ready[state] = new_time;
            if (new_time == time) {
                for (int entry = head[state]; entry != -1; entry = next[entry]) {
                    enqueue(entry / 2);
                }
            }
        };
        // Queue closure handles any chain or cycle with zero running time at
        // this departure tick. Each reachable connection enters at most once.
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const int cid = queue[cursor];
            const int stop = arcs[4 * cid + 2];
            const int arrival = arrivals[cid];
            if (stop == goal) best = std::min(best, arrival);
            relax(stop, arrival + transfers[stop]);
            relax(arcs[4 * cid + 3], arrival);
        }
        for (int state : touched) head[state] = -1;
        left = right;
    }
    return best;
}
