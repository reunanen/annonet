/*
    This example shows how to train a semantic segmentation net using images
    annotated in the "anno" program (see https://github.com/reunanen/anno).

    Instructions:
    1. Use anno to label some data.
    2. Build the annonet_train program.
    3. Run:
       ./annonet_train /path/to/anno/data
    4. Wait while the network is being trained.
    5. Build the annonet_infer example program.
    6. Run:
       ./annonet_infer /path/to/anno/data

    This part of the inference code is here in a separate header so that it's
    easy to embed even in actual applications.
*/

#include "annonet_infer.h"
#include "annonet.h"
#include <dlib/dnn.h>
#include "tiling/dlib-wrapper.h"
#include "tuc/include/tuc/numeric.hpp"
#include <unordered_set>

size_t tensor_index(const dlib::tensor& t, long sample, long k, long row, long column)
{
    // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
    return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
}

void annonet_infer(
    NetPimpl::RuntimeNet& net,
    const NetPimpl::input_type& input_image,
    dlib::matrix<uint16_t>& result_image,
    const std::vector<double>& gains,
    const std::vector<double>& detection_levels,
    const tiling::parameters& tiling_parameters,
    annonet_infer_temp& temp
)
{
    const std::vector<tiling::dlib_tile> tiles = tiling::get_tiles(input_image.nc(), input_image.nr(), tiling_parameters);

    bool first_tile = true;

    for (const tiling::dlib_tile& tile : tiles) {
        const dlib::point tile_center(tile.full_rect.left() + tile.full_rect.width() / 2, tile.full_rect.top() + tile.full_rect.height() / 2);

        const int recommended_tile_width = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.width());
        const int recommended_tile_height = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.height());
        const int recommended_tile_left = tile_center.x() - recommended_tile_width / 2;
        const int recommended_tile_top = tile_center.y() - recommended_tile_height / 2;

        assert(static_cast<unsigned long>(recommended_tile_width) >= tile.full_rect.width());
        assert(static_cast<unsigned long>(recommended_tile_height) >= tile.full_rect.height());

        const dlib::rectangle actual_tile_rect = dlib::rectangle(recommended_tile_left, recommended_tile_top, recommended_tile_left + recommended_tile_width - 1, recommended_tile_top + recommended_tile_height - 1);

        assert(actual_tile_rect.width() == recommended_tile_width);
        assert(actual_tile_rect.height() == recommended_tile_height);

        const int actual_tile_width = actual_tile_rect.width();
        const int actual_tile_height = actual_tile_rect.height();

        const dlib::rectangle actual_tile_centered_rect = dlib::centered_rect(tile_center, actual_tile_width, actual_tile_height);
        assert(actual_tile_rect == actual_tile_centered_rect);

        const dlib::chip_details chip_details(actual_tile_rect, dlib::chip_dims(actual_tile_height, actual_tile_width));
        dlib::extract_image_chip(input_image, chip_details, temp.input_tile, dlib::interpolate_bilinear());

        if (!dlib::rectangle(input_image.nc(), input_image.nr()).contains(chip_details.rect)) {
            const dlib::rectangle inside(-chip_details.rect.tl_corner(), get_rect(input_image).br_corner() - chip_details.rect.tl_corner());
            outpaint(dlib::image_view<NetPimpl::input_type>(temp.input_tile), inside);
        }

        const auto& output_tensor = net.Forward(temp.input_tile);

        if (first_tile) {
            temp.blended_output_tensor.set_size(1, output_tensor.k(), input_image.nr(), input_image.nc());
            std::fill(temp.blended_output_tensor.begin(), temp.blended_output_tensor.end(), 0.f);
            first_tile = false;
        }
        else {
            DLIB_CASSERT(output_tensor.k() == temp.blended_output_tensor.k());
        }

        const long long class_count = output_tensor.k();

        const float* in = output_tensor.host();
        float* out = temp.blended_output_tensor.host();

        const auto get_t = [](long long coordinate, long long first_possible_value, long long first_in_value, long long last_in_value, long long last_possible_value) {
            assert(coordinate >= first_possible_value);
            assert(coordinate <= last_possible_value);
            if (coordinate < first_in_value) {
                return (coordinate - first_possible_value) / static_cast<double>(first_in_value - first_possible_value);
            }
            else if (coordinate > last_in_value) {
                return (last_possible_value - coordinate) / static_cast<double>(last_possible_value - last_in_value);
            }
            else {
                return 1.0;
            }
        };

        for (long long y = 0, blended_y = actual_tile_rect.top(), nr = output_tensor.nr(); y < nr; ++y, ++blended_y) {

            if (blended_y < tile.full_rect.top() || blended_y > tile.full_rect.bottom()) {
                continue;
            }

            if (blended_y < 0 || blended_y >= input_image.nr()) {
                continue;
            }

            for (long long x = 0, blended_x = actual_tile_rect.left(), nc = output_tensor.nc(); x < nc; ++x, ++blended_x) {

                if (blended_x < tile.full_rect.left() || blended_x > tile.full_rect.right()) {
                    continue;
                }

                if (blended_x < 0 || blended_x >= input_image.nc()) {
                    continue;
                }

                assert(tile.full_rect.contains(blended_x, blended_y));

                const auto pixel_requires_blending = !tile.unique_rect.contains(blended_x, blended_y);

                for (long long k = 0; k < class_count; ++k) {

                    const auto& in_index = tensor_index(output_tensor, 0, k, y, x);
                    const auto& out_index = tensor_index(temp.blended_output_tensor, 0, k, blended_y, blended_x);

                    if (pixel_requires_blending) {
                        assert(tiles.size() > 1);
                        const auto th = get_t(blended_x, tile.full_rect.left(), tile.unique_rect.left(), tile.unique_rect.right(), tile.full_rect.right());
                        const auto tv = get_t(blended_y, tile.full_rect.top(), tile.unique_rect.top(), tile.unique_rect.bottom(), tile.full_rect.bottom());
                        assert(th < 1.0 || tv < 1.0);
                        const auto t = th * tv;
                        assert(fabs(tuc::lerp(0.0, th, tv) - t) < 1e-10);
                        assert(fabs(tuc::lerp(0.0, tv, th) - t) < 1e-10);
                        out[out_index] += t * in[in_index];
                    }
                    else {
                        // TODO: it might possibly be a tad more efficient to use a series of memcpy operations (one for each row)
                        //       (especially when tiles.size() == 1, and no blending whatsoever is needed)
                        assert(out[out_index] == 0.f);
                        out[out_index] = in[in_index];
                    }
                }
            }
        }
    }

    result_image.set_size(temp.blended_output_tensor.nr(), temp.blended_output_tensor.nc());

    // The index of the largest output for each element is the label.
    float* out = temp.blended_output_tensor.host();
    const auto find_label = [&](long r, long c)
    {
        uint16_t label = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
        float max_value = -std::numeric_limits<float>::infinity();
        for (long k = 0; k < temp.blended_output_tensor.k(); ++k)
        {
            const double gain = gains.empty() ? 0.0 : gains[k];
            const float value = out[tensor_index(temp.blended_output_tensor, 0, k, r, c)] + gain;
            if (value > max_value)
            {
                label = static_cast<uint16_t>(k);
                max_value = value;
            }
        }
        return label;
    };

    const bool use_detection_level = std::any_of(detection_levels.begin(), detection_levels.end(),
        [](const double value) {
            assert(value >= 0.0);
            return value > 0.0;
        });

    if (use_detection_level) {
        temp.detection_seeds.clear();
    }

    for (long r = 0, nr = temp.blended_output_tensor.nr(); r < nr; ++r)
    {
        for (long c = 0, nc = temp.blended_output_tensor.nc(); c < nc; ++c)
        {
            // The index of the largest output for this element is the label.
            const auto label = find_label(r, c);
            result_image(r, c) = label;

            if (use_detection_level && label > 0) {
                const float clean_output = out[tensor_index(temp.blended_output_tensor, 0, 0, r, c)];
                const float label_output = out[tensor_index(temp.blended_output_tensor, 0, label, r, c)];

                if (label_output - clean_output > detection_levels[label] - detection_levels[0]) {
                    temp.detection_seeds.emplace_back(r, c);
                }
            }
        }
    }

    if (use_detection_level) {
        const unsigned long connected_blob_count = dlib::label_connected_blobs(result_image, dlib::zero_pixels_are_background(), dlib::neighbors_8(), dlib::connected_if_equal(), temp.connected_blobs);

        std::unordered_set<unsigned int> detected_blobs;

        for (const dlib::point& point : temp.detection_seeds) {
            const unsigned int blob = temp.connected_blobs(point.y(), point.x());
            detected_blobs.insert(blob);
        }

        const long nr = input_image.nr();
        const long nc = input_image.nc();

        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                const unsigned int blob = temp.connected_blobs(r, c);
                if (blob > 0) {
                    if (detected_blobs.find(blob) == detected_blobs.end()) {
                        result_image(r, c) = 0;
                    }
                }
            }
        }
    }
}
