#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Forward messages and transition matrices belong to the immutable Python law.
// Uniforms are supplied by the caller; this routine never advances an RNG.
extern "C" void chain_draw(
    int draws, int slots, const int* sizes, const double* const* forward,
    const double* const* transitions, const double* uniforms, intptr_t* result) {
    for (int slot = slots - 1; slot >= 0; --slot) {
        const int states = sizes[slot];
        const int next_states = slot + 1 == slots ? 1 : sizes[slot + 1];
        std::vector<int> first(next_states, -1), following(draws, -1);
        for (int row = 0; row < draws; ++row) {
            int next = slot + 1 == slots ? 0 : result[row * slots + slot + 1];
            following[row] = first[next];
            first[next] = row;
        }
        std::vector<double> weights(states), cumulative(states);
        for (int next = 0; next < next_states; ++next) {
            if (first[next] == -1) continue;
            double peak = -std::numeric_limits<double>::infinity();
            for (int state = 0; state < states; ++state) {
                double weight = forward[slot][state];
                if (slot + 1 < slots)
                    weight += transitions[slot][state * next_states + next];
                weights[state] = weight;
                peak = std::max(peak, weight);
            }
            // The common scale cancels when the cumulative weights normalize.
            double total = 0.;
            for (int state = 0; state < states; ++state) {
                total += std::exp(weights[state] - peak);
                cumulative[state] = total;
            }
            for (double& value : cumulative) value /= total;
            for (int row = first[next]; row != -1; row = following[row]) {
                result[row * slots + slot] = std::upper_bound(
                    cumulative.begin(), cumulative.end(), uniforms[row * slots + slot])
                    - cumulative.begin();
            }
        }
    }
}
