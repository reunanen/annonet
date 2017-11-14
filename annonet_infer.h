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

#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include <dlib/dnn.h>
#include "tiling/dlib-wrapper.h"

void annonet_infer(
    NetPimpl::RuntimeNet& net,
    const NetPimpl::input_type& input_image,
    dlib::matrix<uint16_t>& result_image,
    const std::vector<double>& gains = std::vector<double>(),
    const tiling::parameters& tiling_parameters = tiling::parameters(),
    NetPimpl::input_type& temp_input_tile = NetPimpl::input_type() // Can be supplied to avoid unnecessary memory re-allocations
)
{
    result_image.set_size(input_image.nr(), input_image.nc());

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
        dlib::extract_image_chip(input_image, chip_details, temp_input_tile, dlib::interpolate_bilinear());

        const dlib::matrix<uint16_t> index_label_tile = net(temp_input_tile, gains);

        DLIB_CASSERT(index_label_tile.nr() == temp_input_tile.nr());
        DLIB_CASSERT(index_label_tile.nc() == temp_input_tile.nc());

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
    }
}