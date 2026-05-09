#pragma once

#include "cli/cli_options.h"
#include "utils/image_utils.h"

namespace myimg {

// 应用摄影后期调整
myimg::ImageData apply_photo_adjustments(myimg::ImageData img_data, const CliOptions& opts);

} // namespace myimg
