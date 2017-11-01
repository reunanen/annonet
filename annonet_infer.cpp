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
#include "tiling/dlib-wrapper.h"

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

inline rgb_alpha_pixel index_label_to_rgba_label(uint16_t index_label, const std::vector<AnnoClass>& anno_classes)
{
    const AnnoClass& anno_class = anno_classes[index_label];
    assert(anno_class.index == index_label);
    return anno_class.rgba_label;
}

void index_label_image_to_rgba_label_image(const matrix<uint16_t>& index_label_image, matrix<rgb_alpha_pixel>& rgba_label_image, const std::vector<AnnoClass>& anno_classes)
{
    const long nr = index_label_image.nr();
    const long nc = index_label_image.nc();

    rgba_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            rgba_label_image(r, c) = index_label_to_rgba_label(index_label_image(r, c), anno_classes);
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

struct input_image_type {
    dlib::file file;
    matrix<rgb_pixel> input_image;
    matrix<rgb_alpha_pixel> ground_truth_image;
    std::string error;
};

struct result_image_type {
    std::string filename;
    matrix<rgb_alpha_pixel> label_image;
};

int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        cout << "You call this program like this: " << endl;
        cout << "./annonet_infer /path/to/image/data" << endl;
        cout << endl;
        cout << "You will also need a trained 'annonet.dnn' file. " << endl;
        cout << endl;
        return 1;
    }

    std::string serialized_runtime_net;
    std::string anno_classes_json;
    deserialize("annonet.dnn") >> anno_classes_json >> serialized_runtime_net;

    NetPimpl::RuntimeNet net;
    net.Deserialize(std::istringstream(serialized_runtime_net));

    const std::vector<AnnoClass> anno_classes = parse_anno_classes(anno_classes_json);

    matrix<rgb_pixel> input_tile;
    matrix<uint16_t> index_label_tile_resized;
    matrix<rgb_alpha_pixel> rgba_label_tile;
    matrix<rgb_pixel> result_image;

    auto files = get_images(argv[1]);

    dlib::pipe<dlib::file> full_image_read_requests(files.size());
    for (dlib::file& file : files) {
        full_image_read_requests.enqueue(file);
    }

    dlib::pipe<input_image_type> full_image_read_results(std::thread::hardware_concurrency());

    const auto read_full_image = [](const dlib::file& file)
    {
        input_image_type result;
        try {
            result.file = file;
            load_image(result.input_image, file.full_name());

            const std::string label_filename = file.full_name() + "_mask.png";
            std::ifstream label_file(label_filename, std::ios::binary);
            const bool has_ground_truth = !!label_file;
            if (has_ground_truth) {
                label_file.close();
                load_image(result.ground_truth_image, label_filename);

                if (result.input_image.nr() != result.ground_truth_image.nr() || result.input_image.nc() != result.ground_truth_image.nc()) {
                    result.error = "Label image size mismatch";
                }
            }
        }
        catch (std::exception& e) {
            result.error = e.what();
        }

        return result;
    };

    std::vector<std::thread> full_image_readers;

    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            file file;
            while (full_image_read_requests.dequeue(file)) {
                full_image_read_results.enqueue(read_full_image(file));
            }
        }));
    }

    dlib::pipe<result_image_type> result_image_write_requests(std::thread::hardware_concurrency());
    dlib::pipe<bool> result_image_write_results(files.size());

    std::vector<std::thread> result_image_writers;

    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        result_image_writers.push_back(std::thread([&]() {
            result_image_type result_image;
            while (result_image_write_requests.dequeue(result_image)) {
                save_png(result_image.label_image, result_image.filename);
                result_image_write_results.enqueue(true);
            }
        }));
    }

    tiling::parameters tiling_parameters;
#ifdef DLIB_USE_CUDA
    tiling_parameters.max_tile_width = 640;
    tiling_parameters.max_tile_height = 640;
#else
    // No need for tiling in CPU-only mode
    tiling_parameters.max_tile_width = std::numeric_limits<int>::max();
    tiling_parameters.max_tile_height = std::numeric_limits<int>::max();
#endif

    size_t correct = 0;
    size_t incorrect = 0;

    const auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0, end = files.size(); i < end; ++i)
    {
        std::cout << "\rProcessing image " << (i + 1) << " of " << end << "...";

        input_image_type ii;
        result_image_type result_image;

        full_image_read_results.dequeue(ii);

        if (!ii.error.empty()) {
            throw std::runtime_error(ii.error);
        }

        const auto& input_image = ii.input_image;

        result_image.filename = ii.file.full_name() + "_result.png";
        result_image.label_image.set_size(input_image.nr(), input_image.nc());

        std::vector<dlib::rectangle> tiles = tiling::get_tiles(input_image.nc(), input_image.nr(), tiling_parameters);

        for (const dlib::rectangle& tile : tiles) {
            const long top = tile.top();
            const long left = tile.left();
            const long bottom = tile.bottom();
            const long right = tile.right();
            input_tile.set_size(tile.height(), tile.width());
            for (long y = top; y <= bottom; ++y) {
                for (long x = left; x <= right; ++x) {
                    input_tile(y - top, x - left) = input_image(y, x);
                }
            }
            const matrix<uint16_t> index_label_tile = net(input_tile);
            index_label_tile_resized.set_size(input_tile.nr(), input_tile.nc());
            resize_image(index_label_tile, index_label_tile_resized, interpolate_nearest_neighbor());
            index_label_image_to_rgba_label_image(index_label_tile_resized, rgba_label_tile, anno_classes);
            const long offset_y = top;
            const long offset_x = left;
            for (long tile_y = 0; tile_y < rgba_label_tile.nr(); ++tile_y) {
                for (long tile_x = 0; tile_x < rgba_label_tile.nc(); ++tile_x) {
                    result_image.label_image(tile_y + offset_y, tile_x + offset_x) = rgba_label_tile(tile_y, tile_x);
                }
            }
        }

        if (ii.ground_truth_image.size() > 0) {
            const long nr = result_image.label_image.nr();
            const long nc = result_image.label_image.nr();
            for (size_t r = 0; r < nr; ++r) {
                for (size_t c = 0; c < nc; ++c) {
                    const dlib::rgb_alpha_pixel ground_truth_value = ii.ground_truth_image(r, c);
                    if (ground_truth_value == rgba_ignore_label) {
                        ; // skip the pixel
                    }
                    else {
                        const dlib::rgb_alpha_pixel inference_value = result_image.label_image(r, c);
                        if (inference_value == ground_truth_value) {
                            ++correct;
                        }
                        else {
                            ++incorrect;
                        }
                    }
                }
            }
        }

        result_image_write_requests.enqueue(result_image);
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "\nAll " << files.size() << " images processed in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0 << " seconds!" << std::endl;

    for (size_t i = 0, end = files.size(); i < end; ++i) {
        bool ok;
        result_image_write_results.dequeue(ok);
    }

    std::cout << "All result images written!" << std::endl;

    full_image_read_requests.disable();
    result_image_write_requests.disable();

    for (std::thread& image_reader : full_image_readers) {
        image_reader.join();
    }
    for (std::thread& image_writer : result_image_writers) {
        image_writer.join();
    }

    if (correct > 0 || incorrect > 0) {
        std::cout << "Accuracy = " << 100.0 * correct / (correct + incorrect) << " % (correct = " << correct << ", incorrect = " << incorrect << ")" << std::endl;
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

