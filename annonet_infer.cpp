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
*/

#include "annonet.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_saver/save_png.h>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

const AnnoClass& find_anno_class(const uint16_t& index_label)
{
    return find_anno_class(
        [&index_label](const AnnoClass& anno_class) {
            return index_label == anno_class.index;
        }
    );
}

inline rgb_pixel index_label_to_rgb_label(uint16_t index_label)
{
    return find_anno_class(index_label).rgb_label;
}

void index_label_image_to_rgb_label_image(const matrix<uint16_t>& index_label_image, matrix<rgb_pixel>& rgb_label_image)
{
    const long nr = index_label_image.nr();
    const long nc = index_label_image.nc();

    rgb_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            rgb_label_image(r, c) = index_label_to_rgb_label(index_label_image(r, c));
        }
    }
}

// ----------------------------------------------------------------------------------------

std::vector<file> get_images(
    const std::string& folder
)
{
    const std::vector<file> files = get_files_in_directory_tree(folder,
        [](const file& name) {
            if (match_ending("_mask.png")(name)) {
                return false;
            }
            if (match_ending("_result.png")(name)) {
                return false;
            }
            return match_ending(".jpeg")(name)
                || match_ending(".jpg")(name)
                || match_ending(".png")(name);
        });

    return files;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        cout << "You call this program like this: " << endl;
        cout << "./dnn_semantic_segmentation_anno_ex /path/to/image/data" << endl;
        cout << endl;
        cout << "You will also need a trained 'annonet.dnn' file. " << endl;
        cout << endl;
        return 1;
    }

    anet_type net;
    deserialize("annonet.dnn") >> net;

    matrix<rgb_pixel> input_image, input_tile;
    matrix<uint16_t> index_label_tile_resized;
    matrix<rgb_pixel> rgb_label_image, rgb_label_tile;
    matrix<rgb_pixel> result_image;

    const auto files = get_images(argv[1]);

    const int max_tile_width = 1023;
    const int max_tile_height = 1023;

    for (size_t i = 0, end = files.size(); i < end; ++i)
    {
        const file& file = files[i];
        std::cout << "\rProcessing image " << (i + 1) << " of " << end << "...";
        load_image(input_image, file.full_name());

        rgb_label_image.set_size(input_image.nr(), input_image.nc());

        const auto find_tile_start_position = [](long center, long max_tile_dim) {
            long start_position = center;
            while (start_position > 0) {
                start_position -= max_tile_dim;
            }
            if (start_position <= -max_tile_dim / 2) {
                start_position += max_tile_dim / 2;
            }
            assert(start_position > -max_tile_dim / 2);
            assert(start_position <= 0);
            return start_position;
        };

        const long center_row = input_image.nr() / 2;
        const long center_col = input_image.nc() / 2;
        long first_tile_start_row = find_tile_start_position(center_row, max_tile_height);
        long first_tile_start_col = find_tile_start_position(center_col, max_tile_width);

        for (long tile_start_row = first_tile_start_row; tile_start_row < input_image.nr(); tile_start_row += max_tile_height) {
            for (long tile_start_col = first_tile_start_col; tile_start_col < input_image.nc(); tile_start_col += max_tile_width) {
                const long top = std::max(tile_start_row, 0L);
                const long left = std::max(tile_start_col, 0L);
                const long bottom = std::min(tile_start_row + max_tile_height, input_image.nr()) - 1;
                const long right = std::min(tile_start_col + max_tile_width, input_image.nc()) - 1;
                input_tile.set_size(bottom - top + 1, right - left + 1);
                for (long y = top; y <= bottom; ++y) {
                    for (long x = left; x <= right; ++x) {
                        input_tile(y - top, x - left) = input_image(y, x);
                    }
                }
                const matrix<uint16_t>& index_label_tile = net(input_tile);
                index_label_tile_resized.set_size(input_tile.nr(), input_tile.nc());
                resize_image(index_label_tile, index_label_tile_resized, interpolate_nearest_neighbor());
                index_label_image_to_rgb_label_image(index_label_tile_resized, rgb_label_tile);
                const long offset_y = top;
                const long offset_x = left;
                for (long tile_y = 0; tile_y < rgb_label_tile.nr(); ++tile_y) {
                    for (long tile_x = 0; tile_x < rgb_label_tile.nc(); ++tile_x) {
                        rgb_label_image(tile_y + offset_y, tile_x + offset_x) = rgb_label_tile(tile_y, tile_x);
                    }
                }
            }
        }

        save_png(rgb_label_image, file.full_name() + "_result.png");
    }

    std::cout << "\nAll " << files.size() << " images processed!" << std::endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

