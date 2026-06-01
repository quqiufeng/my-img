#include "pipeline/photo_adjustment.h"
#include "cli/cli_options.h"
#include "utils/image_adjust.h"
#include "utils/lut_loader.h"
#include "utils/dehaze.h"
#include "utils/log.h"
#include <torch/torch.h>
#include <sstream>

namespace myimg {

ImageData apply_photo_adjustments(ImageData img_data, const CliOptions& opts) {
    // Check if any adjustments are needed
    if (opts.temperature != 0.0f || opts.brightness != 0.0f ||
        opts.contrast != 0.0f || opts.saturation != 0.0f ||
        opts.exposure != 0.0f || opts.highlights != 0.0f ||
        opts.shadows != 0.0f || opts.auto_enhance ||
        !opts.curves.empty() ||
        opts.sharpen_amount > 0.0f || opts.denoise_strength > 0.0f ||
        opts.luminance_denoise_strength > 0.0f || opts.color_denoise_strength > 0.0f ||
        opts.whiten_strength > 0.0f || opts.skin_smooth_strength > 0.0f ||
        !opts.skin_tone.empty() || opts.skin_even_strength > 0.0f ||
        !opts.preset.empty() ||
        opts.vignette_strength > 0.0f ||
        !opts.radial_filter.empty() ||
        !opts.graduated_filter.empty() ||
        !opts.lut_path.empty() ||
        opts.dehaze_strength > 0.0f ||
        opts.vibrance != 0.0f || opts.clarity > 0.0f ||
        opts.split_tone_strength > 0.0f ||
        opts.tint != 0.0f || opts.auto_white_balance ||
        opts.blacks != 0.0f || opts.whites != 0.0f ||
        !opts.brightness_curves.empty() ||
        !opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty()) {
        
        auto tensor = myimg::image_data_to_tensor(img_data);
        
        if (opts.auto_enhance) {
            tensor = myimg::auto_enhance(tensor);
        } else {
            if (opts.temperature != 0.0f) tensor = myimg::adjust_temperature(tensor, opts.temperature);
            if (opts.brightness != 0.0f) tensor = myimg::adjust_brightness(tensor, opts.brightness);
            if (opts.contrast != 0.0f) tensor = myimg::adjust_contrast(tensor, opts.contrast);
            if (opts.saturation != 0.0f) tensor = myimg::adjust_saturation(tensor, opts.saturation);
            if (opts.exposure != 0.0f) tensor = myimg::adjust_exposure(tensor, opts.exposure);
            if (opts.highlights != 0.0f) tensor = myimg::adjust_highlights(tensor, opts.highlights);
            if (opts.shadows != 0.0f) tensor = myimg::adjust_shadows(tensor, opts.shadows);
        }
        
        if (opts.denoise_strength > 0.0f) {
            tensor = myimg::denoise(tensor, opts.denoise_strength);
        }
        if (opts.luminance_denoise_strength > 0.0f) {
            tensor = myimg::luminance_denoise(tensor, opts.luminance_denoise_strength);
        }
        if (opts.color_denoise_strength > 0.0f) {
            tensor = myimg::color_denoise(tensor, opts.color_denoise_strength);
        }
        if (opts.sharpen_amount > 0.0f) {
            tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, opts.sharpen_radius, opts.sharpen_threshold);
        }
        if (opts.smart_sharpen_strength > 0.0f) {
            tensor = myimg::smart_sharpen(tensor, opts.smart_sharpen_strength, opts.smart_sharpen_radius);
        }
        if (opts.edge_sharpen_amount > 0.0f) {
            tensor = myimg::edge_sharpen(tensor, opts.edge_sharpen_amount, opts.edge_sharpen_radius, opts.edge_sharpen_threshold);
        }
        
        if (!opts.curves.empty()) {
            tensor = myimg::apply_curves(tensor, opts.curves);
        }
        
        if (!opts.brightness_curves.empty()) {
            tensor = myimg::apply_brightness_curves(tensor, opts.brightness_curves);
        }
        
        if (!opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty()) {
            tensor = myimg::apply_channel_curves(tensor, opts.r_curves, opts.g_curves, opts.b_curves);
        }
        
        if (!opts.preset.empty()) {
            tensor = myimg::apply_preset(tensor, opts.preset);
        }
        
        if (opts.vignette_strength > 0.0f) {
            tensor = myimg::vignette(tensor, opts.vignette_strength, opts.vignette_radius);
        }
        
        if (!opts.radial_filter.empty()) {
            std::stringstream ss(opts.radial_filter);
            float cx, cy, radius, exp_val, cont_val, sat_val;
            char comma;
            ss >> cx >> comma >> cy >> comma >> radius >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
            tensor = myimg::radial_filter(tensor, cx, cy, radius, exp_val, cont_val, sat_val);
        }
        
        if (!opts.graduated_filter.empty()) {
            std::stringstream ss(opts.graduated_filter);
            float angle, pos, width, exp_val, cont_val, sat_val;
            char comma;
            ss >> angle >> comma >> pos >> comma >> width >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
            tensor = myimg::graduated_filter(tensor, angle, pos, width, exp_val, cont_val, sat_val);
        }
        
        if (!opts.lut_path.empty()) {
            myimg::LUT3D lut;
            if (lut.load_from_file(opts.lut_path)) {
                tensor = lut.apply(tensor);
            }
        }
        
        if (opts.dehaze_strength > 0.0f) {
            tensor = myimg::dehaze(tensor, opts.dehaze_strength);
        }
        
        if (opts.vibrance != 0.0f) {
            tensor = myimg::adjust_vibrance(tensor, opts.vibrance);
        }
        
        if (opts.clarity > 0.0f) {
            tensor = myimg::enhance_clarity(tensor, opts.clarity);
        }
        
        if (opts.split_tone_strength > 0.0f) {
            tensor = myimg::split_tone(tensor, opts.split_tone_highlights, opts.split_tone_shadows, opts.split_tone_strength);
        }
        
        if (opts.tint != 0.0f) {
            tensor = myimg::adjust_tint(tensor, opts.tint);
        }
        
        if (opts.auto_white_balance) {
            tensor = myimg::auto_white_balance(tensor);
        }
        
        if (opts.blacks != 0.0f || opts.whites != 0.0f) {
            tensor = myimg::adjust_levels(tensor, opts.blacks, opts.whites);
        }
        
        if (opts.whiten_strength > 0.0f) {
            tensor = myimg::whiten(tensor, opts.whiten_strength);
        }
        if (opts.skin_smooth_strength > 0.0f) {
            tensor = myimg::skin_smooth(tensor, opts.skin_smooth_strength);
        }
        
        if (!opts.skin_tone.empty()) {
            tensor = myimg::skin_tone_match(tensor, opts.skin_tone, opts.skin_tone_strength);
        }
        if (opts.skin_even_strength > 0.0f) {
            tensor = myimg::skin_tone_even(tensor, opts.skin_even_strength);
        }
        
        img_data = myimg::tensor_to_image_data(tensor);
    }
    
    return img_data;
}


} // namespace myimg
