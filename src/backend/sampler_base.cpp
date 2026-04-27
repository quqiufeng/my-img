#include "backend/sampler_base.h"

namespace myimg {

torch::Tensor Sampler::get_sigmas(int steps, const std::string& scheduler) {
    // Simple linear schedule for now
    return torch::linspace(1.0f, 0.0f, steps);
}

} // namespace myimg
