/*
    This example shows how to train a semantic segmentation net using images
    annotated in the "anno" program (see https://github.com/reunanen/anno).

    Instructions:
    1. Use anno to label some data (use the "things" mode).
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
    std::unordered_map<std::string, SegmentationNetPimpl::RuntimeNet>& segmentation_nets_by_classlabel,
    int segmentation_target_size,
    const NetPimpl::input_type& input_image,
    std::vector<instance_segmentation_result>& results,
    const std::vector<double>& gains,
    const tiling::parameters& tiling_parameters,
    annonet_infer_temp& temp
)
{
    const std::vector<tiling::dlib_tile> tiles = tiling::get_tiles(input_image.nc(), input_image.nr(), tiling_parameters);

    for (const tiling::dlib_tile& tile : tiles) {

        const dlib::point tile_center(tile.full_rect.left() + tile.full_rect.width() / 2, tile.full_rect.top() + tile.full_rect.height() / 2);

#if 0
        const int recommended_tile_width = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.width());
        const int recommended_tile_height = NetPimpl::RuntimeNet::GetRecommendedInputDimension(tile.full_rect.height());
#endif
        const int recommended_tile_width = tile.full_rect.width();
        const int recommended_tile_height = tile.full_rect.height();
        const int recommended_tile_left = tile_center.x() - recommended_tile_width / 2;
        const int recommended_tile_top = tile_center.y() - recommended_tile_height / 2;

        assert(static_cast<unsigned long>(recommended_tile_width) >= tile.full_rect.width());
        assert(static_cast<unsigned long>(recommended_tile_height) >= tile.full_rect.height());

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

        const auto tile_labels = net(temp.input_tile, gains);

        const long valid_left_in_image = actual_tile.non_overlapping_rect.left();
        const long valid_top_in_image = actual_tile.non_overlapping_rect.top();
        const long valid_left_in_tile = actual_tile.non_overlapping_rect.left() - actual_tile.full_rect.left();
        const long valid_top_in_tile = actual_tile.non_overlapping_rect.top() - actual_tile.full_rect.top();

        NetPimpl::input_type instance_segmentation_input;
        dlib::matrix<uint16_t> resized_instance_segmentation_output;

        for (const auto& tile_label : tile_labels) {
            auto image_label = tile_label;
            const auto shift_to_image_coordinates = [](const auto tile_coordinate, const auto offset) {
                return tile_coordinate + offset;
            };
            image_label.rect.set_left  (shift_to_image_coordinates(image_label.rect.left  (), valid_left_in_image - valid_left_in_tile));
            image_label.rect.set_right (shift_to_image_coordinates(image_label.rect.right (), valid_left_in_image - valid_left_in_tile));
            image_label.rect.set_top   (shift_to_image_coordinates(image_label.rect.top   (), valid_top_in_image  - valid_top_in_tile ));
            image_label.rect.set_bottom(shift_to_image_coordinates(image_label.rect.bottom(), valid_top_in_image  - valid_top_in_tile ));

            instance_segmentation_result result;
            result.mmod_rect = image_label;

            const auto i = segmentation_nets_by_classlabel.find(image_label.label);
            if (i != segmentation_nets_by_classlabel.end()) {
                // do instance segmentation
                const auto cropping_rect = get_cropping_rect(result.mmod_rect.rect);
                const dlib::chip_details chip_details(cropping_rect, dlib::chip_dims(segmentation_target_size, segmentation_target_size));
                extract_image_chip(input_image, chip_details, instance_segmentation_input, dlib::interpolate_bilinear());
                const auto instance_segmentation_output = i->second(instance_segmentation_input/*, gains*/);
                
                resized_instance_segmentation_output.set_size(
                    static_cast<int>(chip_details.rect.height()),
                    static_cast<int>(chip_details.rect.width())
                );
                dlib::resize_image(instance_segmentation_output, resized_instance_segmentation_output);

                result.segmentation_mask.set_size(input_image.nr(), input_image.nc());
                result.segmentation_mask = 0;

                for (int r = 0; r < resized_instance_segmentation_output.nr(); ++r)
                {
                    for (int c = 0; c < resized_instance_segmentation_output.nc(); ++c)
                    {
                        const auto output = resized_instance_segmentation_output(r, c);
                        if (resized_instance_segmentation_output(r, c))
                        {
                            const auto y = chip_details.rect.top() + r;
                            const auto x = chip_details.rect.left() + c;
                            if (y >= 0 && y < result.segmentation_mask.nr() && x >= 0 && x < result.segmentation_mask.nc())
                            {
                                // TODO: get the actual net output intensity ("confidence")
                                result.segmentation_mask(y, x) = 255;
                            }
                        }
                    }
                }
            }

            results.push_back(result);
        }
    }
}
