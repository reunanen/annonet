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

#include "cxxopts/include/cxxopts.hpp"
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_saver/save_png.h>
#include "tiling/dlib-wrapper.h"

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

struct gain_factor
{
    uint16_t class_index = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    double gain = 1.0;
};

gain_factor parse_gain_factor(const std::string& gain_factor_from_command_line)
{
    const auto colon_pos = gain_factor_from_command_line.find(':');
    if (colon_pos == std::string::npos || colon_pos < 1 || colon_pos >= gain_factor_from_command_line.length() - 1) {
        throw std::runtime_error("The gain factors must be supplied in the format index:factor (e.g., 1:1.5)");
    }
    gain_factor gain_factor;
    gain_factor.class_index = std::stoul(gain_factor_from_command_line.substr(0, colon_pos));
    gain_factor.gain = std::stod(gain_factor_from_command_line.substr(colon_pos + 1));
    return gain_factor;
}

std::vector<double> parse_gain_factors(const std::vector<std::string>& gain_factors_from_command_line, uint16_t class_count)
{
    std::vector<double> gain_factors(class_count, 1.0);

    for (const auto gain_factor_from_command_line : gain_factors_from_command_line) {
        const gain_factor gain_factor = parse_gain_factor(gain_factor_from_command_line);
        if (gain_factor.class_index >= class_count) {
            std::ostringstream error;
            error << "Can't set gain factor for index " << gain_factor.class_index << " when there are only " << class_count << " classes";
            throw std::runtime_error(error.str());
        }
        gain_factors[gain_factor.class_index] = gain_factor.gain;
    }

    return gain_factors;
}

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

    const size_t class_count = anno_classes.size();

    std::ostringstream max_value_string;
    max_value_string << max_value;

    std::ostringstream max_class_string;
    max_class_string << class_count - 1;

    const std::string truth_label = "truth";
    const std::string predicted_label = "predicted";
    const std::string precision_label = "precision";
    const std::string recall_label = "recall";
    const std::string shortest_max_precision_string = "100 %";

    const size_t max_value_length = max_value_string.str().length();
    const size_t value_column_width = std::max(shortest_max_precision_string.length() + 1, max_value_length + 2);

    const size_t max_class_length = max_class_string.str().length();
    const size_t class_column_width = max_class_length + 3;

    const size_t recall_column_width = recall_label.length() + 4;

    { // Print the 'predicted' label
        const size_t padding = truth_label.length() + class_column_width + value_column_width * class_count / 2 + predicted_label.length() / 2;
        std::cout << std::setw(padding) << std::right << predicted_label << std::endl;
    }

    // Print class headers
    std::cout << std::setw(truth_label.length() + class_column_width) << ' ';
    for (const auto& anno_class : anno_classes) {
        std::cout << std::right << std::setw(value_column_width) << anno_class.index;
    }
    std::cout << std::setw(recall_column_width) << std::right << recall_label << std::endl;

    // Print the confusion matrix itself
    std::vector<size_t> total_predicted(class_count);
    size_t total_correct = 0;
    size_t total = 0;

    for (size_t ground_truth_index = 0; ground_truth_index < class_count; ++ground_truth_index) {
        DLIB_CASSERT(ground_truth_index == anno_classes[ground_truth_index].index);
        std::cout << std::setw(truth_label.length());
        if (ground_truth_index == (class_count - 1) / 2) {
            std::cout << truth_label;
        }
        else {
            std::cout << ' ';
        }
        std::cout << std::right << std::setw(class_column_width) << ground_truth_index;
        size_t total_ground_truth = 0;
        for (size_t predicted_index = 0; predicted_index < class_count; ++predicted_index) {
            const auto& predicted = confusion_matrix[ground_truth_index][predicted_index];
            std::cout << std::right << std::setw(value_column_width) << predicted;
            total_predicted[predicted_index] += predicted;
            total_ground_truth += predicted;
            if (predicted_index == ground_truth_index) {
                total_correct += predicted;
            }
            total += predicted;
        }
        std::cout << std::setw(recall_column_width) << std::fixed << std::setprecision(2);
        std::cout << confusion_matrix[ground_truth_index][ground_truth_index] * 100.0 / total_ground_truth << " %";
        std::cout << std::endl;
    }

    // Print precision
    assert(truth_label.length() + class_column_width <= precision_label.length());
    const auto precision_accuracy = std::min(static_cast<size_t>(2), value_column_width - shortest_max_precision_string.length() - 1);
    std::cout << std::setw(truth_label.length() + class_column_width) << precision_label << "  ";
    for (size_t predicted_index = 0; predicted_index < class_count; ++predicted_index) {
        std::cout << std::right << std::setw(value_column_width - 2) << std::fixed << std::setprecision(precision_accuracy);
        if (total_predicted[predicted_index] > 0) {
            std::cout << confusion_matrix[predicted_index][predicted_index] * 100.0 / total_predicted[predicted_index] << " %";
        }
        else {
            std::cout << "-" << "  ";
        }
    }
    std::cout << std::endl;

    // Print accuracy
    std::cout << std::setw(truth_label.length() + class_column_width + class_count * value_column_width) << std::right << "accuracy";
    std::cout << std::right << std::setw(recall_column_width) << std::fixed << std::setprecision(2);
    std::cout << total_correct * 100.0 / total << " %" << std::endl;
}

// ----------------------------------------------------------------------------------------

struct result_image_type {
    std::string filename;
    int original_width = 0;
    int original_height = 0;
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

    cxxopts::Options options("annonet_infer", "Do inference using trained semantic-segmentation networks");

#ifdef DLIB_USE_CUDA
    const std::string default_max_tile_width = "512";
    const std::string default_max_tile_height = "512";
#else
    // in CPU-only mode, we can handle larger tiles
    const std::string default_max_tile_width = "4096";
    const std::string default_max_tile_height = "4096";
#endif

    options.add_options()
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
        ("g,gain-factor", "Supply a class-specific gain factor, for example: 1:1.5", cxxopts::value<std::vector<std::string>>())
        ("w,tile-max-width", "Set max tile width", cxxopts::value<int>()->default_value(default_max_tile_width))
        ("h,tile-max-height", "Set max tile height", cxxopts::value<int>()->default_value(default_max_tile_height))
        ;

    try {
        options.parse_positional("input-directory");
        options.parse(argc, argv);

        cxxopts::check_required(options, { "input-directory" });

        std::cout << "Input directory = " << options["input-directory"].as<std::string>() << std::endl;
    }
    catch (std::exception& e) {
        cerr << e.what() << std::endl;
        cerr << std::endl;
        cerr << options.help() << std::endl;
        return 2;
    }

    double downscaling_factor = 1.0;
    std::string serialized_runtime_net;
    std::string anno_classes_json;
    deserialize("annonet.dnn") >> anno_classes_json >> downscaling_factor >> serialized_runtime_net;

    std::cout << "Deserializing annonet, downscaling factor = " << downscaling_factor << std::endl;

    NetPimpl::RuntimeNet net;
    net.Deserialize(std::istringstream(serialized_runtime_net));

    const std::vector<AnnoClass> anno_classes = parse_anno_classes(anno_classes_json);

    const std::vector<double> gain_factors = parse_gain_factors(options["gain-factor"].as<std::vector<std::string>>(), anno_classes.size());

    assert(gain_factors.size() == anno_classes.size());

    std::cout << "Using gain factors:";
    for (size_t class_index = 0, end = gain_factors.size(); class_index < end; ++class_index) {
        std::cout << " " << class_index << ":" << gain_factors[class_index];
    }
    std::cout << std::endl;

    NetPimpl::input_type input_tile;
    matrix<uint16_t> index_label_tile_resized;

    auto files = find_image_files(options["input-directory"].as<std::string>(), false);

    dlib::pipe<image_filenames> full_image_read_requests(files.size());
    for (const image_filenames& file : files) {
        full_image_read_requests.enqueue(image_filenames(file));
    }

    dlib::pipe<sample> full_image_read_results(std::thread::hardware_concurrency());

    std::vector<std::thread> full_image_readers;

    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            image_filenames image_filenames;
            while (full_image_read_requests.dequeue(image_filenames)) {
                full_image_read_results.enqueue(read_sample(image_filenames, anno_classes, false, downscaling_factor));
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
                resize_label_image(result_image.label_image, result_image.original_width, result_image.original_height);
                index_label_image_to_rgba_label_image(result_image.label_image, rgba_label_image, anno_classes);
                save_png(rgba_label_image, result_image.filename);
                result_image_write_results.enqueue(true);
            }
        }));
    }

    const int min_input_dimension = NetPimpl::TrainingNet::GetRequiredInputDimension();

    tiling::parameters tiling_parameters;
    tiling_parameters.max_tile_width = options["tile-max-width"].as<int>();
    tiling_parameters.max_tile_height = options["tile-max-height"].as<int>();
    tiling_parameters.overlap_x = min_input_dimension;
    tiling_parameters.overlap_y = min_input_dimension;

    DLIB_CASSERT(tiling_parameters.max_tile_width >= min_input_dimension);
    DLIB_CASSERT(tiling_parameters.max_tile_height >= min_input_dimension);

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
        result_image.original_width = sample.original_width;
        result_image.original_height = sample.original_height;

        const std::vector<tiling::dlib_tile> tiles = tiling::get_tiles(input_image.nc(), input_image.nr(), tiling_parameters);

        for (const tiling::dlib_tile& tile : tiles) {
            const dlib::point tile_center(tile.full_rect.left() + tile.full_rect.width() / 2, tile.full_rect.top() + tile.full_rect.height() / 2);
            const int actual_tile_width = std::max(static_cast<int>(tile.full_rect.width()), min_input_dimension);
            const int actual_tile_height = std::max(static_cast<int>(tile.full_rect.height()), min_input_dimension);
            const rectangle actual_tile_rect = centered_rect(tile_center, actual_tile_width, actual_tile_height);
            const chip_details chip_details(actual_tile_rect, chip_dims(actual_tile_height, actual_tile_width));
            extract_image_chip(input_image, chip_details, input_tile, interpolate_bilinear());

            const matrix<uint16_t> index_label_tile = net(input_tile, gain_factors);

            index_label_tile_resized.set_size(input_tile.nr(), input_tile.nc());
            resize_image(index_label_tile, index_label_tile_resized, interpolate_nearest_neighbor());

            const long valid_left_in_image = tile.non_overlapping_rect.left();
            const long valid_top_in_image = tile.non_overlapping_rect.top();
            const long valid_left_in_tile = tile.non_overlapping_rect.left() - tile.full_rect.left() + std::max(0, (min_input_dimension - static_cast<int>(tile.full_rect.width())) / 2);
            const long valid_top_in_tile = tile.non_overlapping_rect.top() - tile.full_rect.top() + std::max(0, (min_input_dimension - static_cast<int>(tile.full_rect.height())) / 2);
            for (long y = 0, valid_tile_height = tile.non_overlapping_rect.height(); y < valid_tile_height; ++y) {
                for (long x = 0, valid_tile_width = tile.non_overlapping_rect.width(); x < valid_tile_width; ++x) {
                    const uint16_t label = index_label_tile_resized(valid_top_in_tile + y, valid_left_in_tile + x);
                    result_image.label_image(valid_top_in_image + y, valid_left_in_image + x) = label;
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
        std::cout << std::endl << "Confusion matrix:" << std::endl;
        print_confusion_matrix(confusion_matrix, anno_classes);
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}