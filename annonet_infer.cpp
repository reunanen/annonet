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
#include <dlib/dnn.h>
#include "tiling/dlib-wrapper.h"
#include <unordered_set>

template <
    typename image_type
>
void outpaint(
    dlib::image_view<image_type>& img,
    dlib::rectangle inside
)
{
    inside = inside.intersect(get_rect(img));
    if (inside.is_empty())
    {
        return;
    }

    for (long r = 0; r < inside.top(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[inside.top()][inside.left()];
        }
        for (long c = inside.left(); c <= inside.right(); ++c) {
            img[r][c] = img[inside.top()][c];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[inside.top()][inside.right()];
        }
    }
    for (long r = inside.top(); r <= inside.bottom(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[r][inside.left()];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[r][inside.right()];
        }
    }
    for (long r = inside.bottom() + 1; r < img.nr(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[inside.bottom()][inside.left()];
        }
        for (long c = inside.left(); c <= inside.right(); ++c) {
            img[r][c] = img[inside.bottom()][c];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[inside.bottom()][inside.right()];
        }
    }

    // TODO: even blur from outside
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
    const bool use_detection_level = std::any_of(detection_levels.begin(), detection_levels.end(),
        [](const double value) {
            assert(value >= 0.0);
            return value > 0.0;
        });

    result_image.set_size(input_image.nr(), input_image.nc());

    if (use_detection_level) {
        temp.detection_seeds.clear();
    }

    const std::vector<tiling::dlib_tile> tiles = tiling::get_tiles(input_image.nc(), input_image.nr(), tiling_parameters);

    for (const tiling::dlib_tile& tile : tiles) {

        const dlib::point tile_center(tile.full_rect.left() + tile.full_rect.width() / 2, tile.full_rect.top() + tile.full_rect.height() / 2);

        const int recommended_tile_width = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.width());
        const int recommended_tile_height = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.height());
        const int recommended_tile_left = tile_center.x() - recommended_tile_width / 2;
        const int recommended_tile_top = tile_center.y() - recommended_tile_height / 2;

        assert(recommended_tile_width >= tile.full_rect.width());
        assert(recommended_tile_height >= tile.full_rect.height());

        tiling::dlib_tile actual_tile;
        actual_tile.full_rect = dlib::rectangle(recommended_tile_left, recommended_tile_top, recommended_tile_left + recommended_tile_width - 1, recommended_tile_top + recommended_tile_height - 1);
        actual_tile.non_overlapping_rect = tile.non_overlapping_rect;

        assert(actual_tile.full_rect.width() == recommended_tile_width);
        assert(actual_tile.full_rect.height() == recommended_tile_height);

        const int actual_tile_width = actual_tile.full_rect.width();
        const int actual_tile_height = actual_tile.full_rect.height();
        const dlib::rectangle actual_tile_rect = dlib::centered_rect(tile_center, actual_tile_width, actual_tile_height);
        const dlib::chip_details chip_details(actual_tile_rect, dlib::chip_dims(actual_tile_height, actual_tile_width));
        dlib::extract_image_chip(input_image, chip_details, temp.input_tile, dlib::interpolate_bilinear());

        if (!dlib::rectangle(input_image.nc(), input_image.nr()).contains(chip_details.rect)) {
            const dlib::rectangle inside(-chip_details.rect.tl_corner(), get_rect(input_image).br_corner() - chip_details.rect.tl_corner());
            outpaint(dlib::image_view<NetPimpl::input_type>(temp.input_tile), inside);
        }

        const dlib::matrix<uint16_t> index_label_tile = net(temp.input_tile, gains);

        DLIB_CASSERT(index_label_tile.nr() == temp.input_tile.nr());
        DLIB_CASSERT(index_label_tile.nc() == temp.input_tile.nc());

        const long valid_left_in_image = actual_tile.non_overlapping_rect.left();
        const long valid_top_in_image = actual_tile.non_overlapping_rect.top();
        const long valid_left_in_tile = actual_tile.non_overlapping_rect.left() - actual_tile.full_rect.left();
        const long valid_top_in_tile = actual_tile.non_overlapping_rect.top() - actual_tile.full_rect.top();
        for (long y = 0, valid_tile_height = actual_tile.non_overlapping_rect.height(); y < valid_tile_height; ++y) {
            for (long x = 0, valid_tile_width = actual_tile.non_overlapping_rect.width(); x < valid_tile_width; ++x) {
                const uint16_t label = index_label_tile(valid_top_in_tile + y, valid_left_in_tile + x);
                result_image(valid_top_in_image + y, valid_left_in_image + x) = label;
            }
        }

        if (use_detection_level) {

            const auto tensor_index = [](const dlib::tensor& t, long sample, long k, long row, long column)
            {
                // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
                return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
            };

            const dlib::tensor& output_tensor = net.GetOutput();

            DLIB_CASSERT(output_tensor.nr() == recommended_tile_height);
            DLIB_CASSERT(output_tensor.nc() == recommended_tile_width);

            const float* const out_data = output_tensor.host();

            for (long y = 0, valid_tile_height = actual_tile.non_overlapping_rect.height(); y < valid_tile_height; ++y) {
                for (long x = 0, valid_tile_width = actual_tile.non_overlapping_rect.width(); x < valid_tile_width; ++x) {
                    const uint16_t label = index_label_tile(valid_top_in_tile + y, valid_left_in_tile + x);
                    if (label > 0) {
                        const float clean_output = out_data[tensor_index(output_tensor, 0, 0, valid_top_in_tile + y, valid_left_in_tile + x)];
                        const float label_output = out_data[tensor_index(output_tensor, 0, label, valid_top_in_tile + y, valid_left_in_tile + x)];
                        if (label_output - clean_output > detection_levels[label] - detection_levels[0]) {
                            temp.detection_seeds.emplace_back(valid_left_in_image + x, valid_top_in_image + y);
                        }                        
                    }
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
