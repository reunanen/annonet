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

// first index: ground truth, second index: predicted
typedef std::vector<std::vector<size_t>> confusion_matrix_type;

void init_confusion_matrix(confusion_matrix_type& confusion_matrix, size_t class_count)
{
    confusion_matrix.resize(class_count);
    for (auto& i : confusion_matrix) {
        i.resize(class_count);
    }
}

void print_confusion_matrix(const confusion_matrix_type& confusion_matrix, const std::vector<AnnoClass>& anno_classes)
{
    size_t max_value = 0;
    for (const auto& ground_truth : confusion_matrix) {
        for (const auto& predicted : ground_truth) {
            max_value = std::max(max_value, predicted);
        }
    }

    std::ostringstream max_value_string;
    max_value_string << max_value;

    const size_t max_value_length = max_value_string.str().length();
    const size_t column_width = std::max(static_cast<size_t>(4), max_value_length + 2);

    std::cout << std::setw(column_width) << ' ';
    for (const auto& anno_class : anno_classes) {
        std::cout << std::right << std::setw(column_width) << anno_class.index;
    }
    std::cout << std::endl;

    for (size_t ground_truth_index = 0, end = confusion_matrix.size(); ground_truth_index < end; ++ground_truth_index) {
        DLIB_CASSERT(ground_truth_index == anno_classes[ground_truth_index].index);
        std::cout << std::right << std::setw(column_width) << ground_truth_index;
        for (const auto& predicted : confusion_matrix[ground_truth_index]) {
            std::cout << std::right << std::setw(column_width) << predicted;
        }
        std::cout << std::endl;
    }

}

// ----------------------------------------------------------------------------------------

struct result_image_type {
    std::string filename;
    matrix<uint16_t> label_image;
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

    matrix<input_pixel_type> input_tile;
    matrix<uint16_t> index_label_tile_resized;

    auto files = find_image_files(argv[1], false);

    dlib::pipe<image_filenames> full_image_read_requests(files.size());
    for (const image_filenames& file : files) {
        full_image_read_requests.enqueue(image_filenames(file));
    }

    dlib::pipe<sample> full_image_read_results(std::thread::hardware_concurrency());

    const auto read_full_image = [&anno_classes](const image_filenames& image_filenames)
    {
        sample result;
        try {
            result.image_filenames = image_filenames;
            load_image(result.input_image, image_filenames.image_filename);

            if (!image_filenames.label_filename.empty()) {
                matrix<rgb_alpha_pixel> ground_truth_image;
                load_image(ground_truth_image, image_filenames.label_filename);

                if (result.input_image.nr() != ground_truth_image.nr() || result.input_image.nc() != ground_truth_image.nc()) {
                    result.error = "Label image size mismatch";
                }
                else {
                    decode_rgba_label_image(ground_truth_image, result, anno_classes);
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
            image_filenames image_filenames;
            while (full_image_read_requests.dequeue(image_filenames)) {
                full_image_read_results.enqueue(read_sample(image_filenames, anno_classes, false));
            }
        }));
    }

    dlib::pipe<result_image_type> result_image_write_requests(std::thread::hardware_concurrency());
    dlib::pipe<bool> result_image_write_results(files.size());

    std::vector<std::thread> result_image_writers;

    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        result_image_writers.push_back(std::thread([&]() {
            result_image_type result_image;
            dlib::matrix<rgb_alpha_pixel> rgba_label_image;
            while (result_image_write_requests.dequeue(result_image)) {
                index_label_image_to_rgba_label_image(result_image.label_image, rgba_label_image, anno_classes);
                save_png(rgba_label_image, result_image.filename);
                result_image_write_results.enqueue(true);
            }
        }));
    }

    tiling::parameters tiling_parameters;
#ifdef DLIB_USE_CUDA
    tiling_parameters.max_tile_width = 640;
    tiling_parameters.max_tile_height = 640;
#else
    // in CPU-only mode, we can handle much larger tiles
    tiling_parameters.max_tile_width = 4096;
    tiling_parameters.max_tile_height = 4096;
#endif

    // first index: ground truth, second index: predicted
    confusion_matrix_type confusion_matrix;
    init_confusion_matrix(confusion_matrix, anno_classes.size());
    size_t ground_truth_count = 0;

    const auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0, end = files.size(); i < end; ++i)
    {
        std::cout << "\rProcessing image " << (i + 1) << " of " << end << "...";

        sample sample;
        result_image_type result_image;

        full_image_read_results.dequeue(sample);

        if (!sample.error.empty()) {
            throw std::runtime_error(sample.error);
        }

        const auto& input_image = sample.input_image;

        result_image.filename = sample.image_filenames.image_filename + "_result.png";
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
            const long offset_y = top;
            const long offset_x = left;
            const long nr = index_label_tile_resized.nr();
            const long nc = index_label_tile_resized.nc();
            for (long tile_y = 0; tile_y < nr; ++tile_y) {
                for (long tile_x = 0; tile_x < nc; ++tile_x) {
                    result_image.label_image(tile_y + offset_y, tile_x + offset_x) = index_label_tile_resized(tile_y, tile_x);
                }
            }
        }

        for (const auto& labeled_points : sample.labeled_points_by_class) {
            const uint16_t ground_truth_value = labeled_points.first;
            for (const dlib::point& point : labeled_points.second) {
                const uint16_t predicted_value = result_image.label_image(point.y(), point.x());
                ++confusion_matrix[ground_truth_value][predicted_value];                    
            }
            ground_truth_count += labeled_points.second.size();
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

    if (ground_truth_count) {
        std::cout << "Confusion matrix:" << std::endl;
        print_confusion_matrix(confusion_matrix, anno_classes);
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

