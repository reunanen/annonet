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
#include "annonet_infer.h"

#include "cxxopts/include/cxxopts.hpp"
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_saver/save_png.h>
#include "tiling/dlib-wrapper.h"

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

struct class_specific_value_type
{
    uint16_t class_index = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    double value = 0.0;
};

class_specific_value_type parse_class_specific_value(const std::string& string_from_command_line)
{
    const auto colon_pos = string_from_command_line.find(':');
    if (colon_pos == std::string::npos || colon_pos < 1 || colon_pos >= string_from_command_line.length() - 1) {
        throw std::runtime_error("The gains must be supplied in the format index:gain (e.g., 1:-0.5)");
    }
    class_specific_value_type class_specific_value;
    class_specific_value.class_index = std::stoul(string_from_command_line.substr(0, colon_pos));
    class_specific_value.value = std::stod(string_from_command_line.substr(colon_pos + 1));
    return class_specific_value;
}

std::vector<double> parse_class_specific_values(const std::vector<std::string>& strings_from_command_line, uint16_t class_count)
{
    std::vector<double> class_specific_values(class_count, 0.0);

    for (const auto string_from_command_line : strings_from_command_line) {
        const auto class_specific_value = parse_class_specific_value(string_from_command_line);
        if (class_specific_value.class_index >= class_count) {
            std::ostringstream error;
            error << "Can't define class-specific value for index " << class_specific_value.class_index << " when there are only " << class_count << " classes";
            throw std::runtime_error(error.str());
        }
        class_specific_values[class_specific_value.class_index] = class_specific_value.value;
    }

    return class_specific_values;
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

struct update_confusion_matrix_per_region_temp_type
{
    dlib::matrix<int> ground_truth_blobs;
    dlib::matrix<int> result_blobs;
};

void update_confusion_matrix_per_region(
    confusion_matrix_type& confusion_matrix_per_region,
    const std::unordered_map<uint16_t, std::deque<dlib::point>>& labeled_points_by_class,
    const dlib::matrix<uint16_t>& ground_truth_label_image,
    const dlib::matrix<uint16_t>& result_label_image,
    update_confusion_matrix_per_region_temp_type& temp
)
{
    if (labeled_points_by_class.empty()) {
        return;
    }

    DLIB_CASSERT(ground_truth_label_image.nr() == result_label_image.nr());
    DLIB_CASSERT(ground_truth_label_image.nc() == result_label_image.nc());

    const unsigned long ground_truth_blob_count = dlib::label_connected_blobs(ground_truth_label_image, zero_pixels_are_background(), neighbors_8(), connected_if_equal(), temp.ground_truth_blobs);
    const unsigned long result_blob_count       = dlib::label_connected_blobs(result_label_image,       zero_pixels_are_background(), neighbors_8(), connected_if_equal(), temp.result_blobs);

    const auto vote_blob_class = [&](int blob_number, const dlib::matrix<int>& blobs) {
        std::unordered_map<uint16_t, size_t> votes_ground_truth;
        std::unordered_map<uint16_t, size_t> votes_predicted;

        const auto find_class_with_most_votes = [](const std::unordered_map<uint16_t, size_t>& votes) {
            if (votes.empty()) {
                return static_cast<uint16_t>(dlib::loss_multiclass_log_per_pixel_::label_to_ignore);
            }
            const auto max_vote = std::max_element(votes.begin(), votes.end(),
                [](const pair<uint16_t, size_t>& vote1, const pair<uint16_t, size_t>& vote2) {
                return vote1.second < vote2.second;
            });
            assert(max_vote != votes.end());
            return max_vote->first;
        };

        for (const auto i : labeled_points_by_class) {
            const auto ground_truth = i.first;
            for (const dlib::point& point : i.second) {
                const auto x = point.x();
                const auto y = point.y();
                if (blobs(y, x) == blob_number) {
                    assert(ground_truth_label_image(y, x) == ground_truth);
                    ++votes_ground_truth[ground_truth];
                    const auto predicted = result_label_image(y, x);
                    ++votes_predicted[predicted];
                }
            }

            // If ground-truth is predominantly non-background, consider predictions to be background only if there are not any other votes.
            // (Rationale: in our world, detections are important - we do not want to ignore any, even if they are small in terms of area.)
            const bool ground_truth_predominantly_non_background = find_class_with_most_votes(votes_ground_truth) != 0;
            const bool predicted_background_only = votes_predicted.size() == 1 && votes_predicted.find(0) != votes_predicted.end();
            if (ground_truth_predominantly_non_background && !predicted_background_only) {
                votes_predicted.erase(0);
            }
        }

        return std::make_pair(find_class_with_most_votes(votes_ground_truth), find_class_with_most_votes(votes_predicted));
    };

    for (unsigned long blob = 0; blob < ground_truth_blob_count; ++blob) {
        const auto v = vote_blob_class(blob, temp.ground_truth_blobs);
        if (v.first != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
            ++confusion_matrix_per_region[v.first][v.second];
        }
    }

    for (unsigned long blob = 0; blob < result_blob_count; ++blob) {
        const auto v = vote_blob_class(blob, temp.result_blobs);
        if (v.first != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
            ++confusion_matrix_per_region[v.first][v.second];
        }
    }
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

    std::ostringstream hardware_concurrency;
    hardware_concurrency << std::thread::hardware_concurrency();

#ifdef DLIB_USE_CUDA
    const std::string default_max_tile_width = "1024";
    const std::string default_max_tile_height = "1024";
#else
    // in CPU-only mode, we can handle larger tiles
    const std::string default_max_tile_width = "4096";
    const std::string default_max_tile_height = "4096";
#endif

    options.add_options()
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
        ("g,gain", "Supply a class-specific gain, for example: 1:-0.5", cxxopts::value<std::vector<std::string>>())
        ("d,detection", "Supply a class-specific detection level that _comes on top of gain_, for example: 1:1.5", cxxopts::value<std::vector<std::string>>())
        ("w,tile-max-width", "Set max tile width", cxxopts::value<int>()->default_value(default_max_tile_width))
        ("h,tile-max-height", "Set max tile height", cxxopts::value<int>()->default_value(default_max_tile_height))
        ("full-image-reader-thread-count", "Set the number of full-image reader threads", cxxopts::value<int>()->default_value(hardware_concurrency.str()))
        ("result-image-writer-thread-count", "Set the number of result-image writer threads", cxxopts::value<int>()->default_value(hardware_concurrency.str()))
        ("just-visualize-rects", "Visualize the processing rectangles for debugging purposes (don't run actual inference)")
        ("write-confidence-maps", "Write confidence maps to disk")
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
    {
        std::istringstream iss(serialized_runtime_net);
        net.Deserialize(iss);
    }

    const std::vector<AnnoClass> anno_classes = parse_anno_classes(anno_classes_json);

    DLIB_CASSERT(anno_classes.size() >= 2);

    const std::vector<double> gains = parse_class_specific_values(options["gain"].as<std::vector<std::string>>(), anno_classes.size());
    const std::vector<double> detection_levels = parse_class_specific_values(options["detection"].as<std::vector<std::string>>(), anno_classes.size());

    assert(gains.size() == anno_classes.size());
    assert(detection_levels.size() == anno_classes.size());

    std::cout << "Using gains:";
    for (size_t class_index = 0, end = gains.size(); class_index < end; ++class_index) {
        std::cout << " " << class_index << ":" << gains[class_index];
    }
    std::cout << std::endl;

    std::cout << "Using detection levels:";
    for (size_t class_index = 0, end = detection_levels.size(); class_index < end; ++class_index) {
        std::cout << " " << class_index << ":" << detection_levels[class_index];
    }
    std::cout << std::endl;

    set_low_priority();

    annonet_infer_temp temp;
    matrix<uint16_t> index_label_tile_resized;

    auto files = find_image_files(options["input-directory"].as<std::string>(), false);

    dlib::pipe<image_filenames_type> full_image_read_requests(files.size());
    for (const auto& file : files) {
        full_image_read_requests.enqueue(image_filenames_type(file));
    }

    const int full_image_reader_count = std::max(1, options["full-image-reader-thread-count"].as<int>());
    const int result_image_writer_count = std::max(1, options["result-image-writer-thread-count"].as<int>());

    dlib::pipe<sample_type> full_image_read_results(full_image_reader_count);

    std::vector<std::thread> full_image_readers;

    for (unsigned int i = 0; i < full_image_reader_count; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            image_filenames_type image_filenames;
            while (full_image_read_requests.dequeue(image_filenames)) {
                full_image_read_results.enqueue(read_sample(image_filenames, anno_classes, false, downscaling_factor));
            }
        }));
    }

    dlib::pipe<result_image_type> result_image_write_requests(result_image_writer_count);
    dlib::pipe<bool> result_image_write_results(files.size());

    std::vector<std::thread> result_image_writers;

    for (unsigned int i = 0; i < result_image_writer_count; ++i) {
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
    confusion_matrix_type confusion_matrix_per_pixel, confusion_matrix_per_region;
    init_confusion_matrix(confusion_matrix_per_pixel, anno_classes.size());
    init_confusion_matrix(confusion_matrix_per_region, anno_classes.size());
    size_t ground_truth_count = 0;

    const auto t0 = std::chrono::steady_clock::now();

    update_confusion_matrix_per_region_temp_type update_confusion_matrix_per_region_temp;

    std::chrono::microseconds total_time_spent_in_actual_inference(0);
    std::chrono::microseconds total_time_spent_in_actual_inference_excluding_first_image(0);
    std::chrono::microseconds max_time_spent_in_actual_inference_per_image_excluding_first_image(0);

    for (size_t i = 0, end = files.size(); i < end; ++i)
    {
        std::cout << "\rProcessing image " << (i + 1) << " of " << end << "...";

        sample_type sample;
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

        const auto t0 = std::chrono::steady_clock::now();

        const auto just_visualize_rects = options.count("just-visualize-rects") > 0;

        if (just_visualize_rects) {
            std::fill(result_image.label_image.begin(), result_image.label_image.end(), 0);
            const std::vector<tiling::dlib_tile> tiles = tiling::get_tiles(sample.input_image.nc(), sample.input_image.nr(), tiling_parameters);
            for (const auto& tile : tiles) {
                dlib::draw_rectangle(result_image.label_image, tile.full_rect, std::min(1ull, anno_classes.size() - 1), 1);
                dlib::draw_rectangle(result_image.label_image, tile.non_overlapping_rect, std::min(2ull, anno_classes.size() - 1), 1);
                dlib::draw_rectangle(result_image.label_image, tile.unique_rect, std::min(3ull, anno_classes.size() - 1), 1);
            }
        }
        else {
            annonet_infer(net, sample.input_image, result_image.label_image, temp, gains, detection_levels, tiling_parameters);

            const auto write_confidence_maps = options.count("write-confidence-maps") > 0;

            if (write_confidence_maps) {
                dlib::matrix<uint8_t> image(temp.blended_output_tensor.nr(), temp.blended_output_tensor.nc());

                const auto tensor_index = [](const dlib::tensor& t, long sample, long k, long row, long column) {
                    // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
                    return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
                };

                const auto data = temp.blended_output_tensor.host();

                for (int k = 0; k < temp.blended_output_tensor.k(); ++k) {
                    for (int r = 0; r < temp.blended_output_tensor.nr(); ++r) {
                        for (int c = 0; c < temp.blended_output_tensor.nc(); ++c) {
                            const float raw_value = data[tensor_index(temp.blended_output_tensor, 0, k, r, c)];
                            const int converted_value = std::max(0, std::min(255, static_cast<int>(std::round(((raw_value + 1.f) / 2.f * 255)))));
                            image(r, c) = static_cast<int>(converted_value);
                        }
                    }

                    save_png(image, sample.image_filenames.image_filename + "_k" + std::to_string(k) + ".png");
                }
            }
        }

        const auto t1 = std::chrono::steady_clock::now();

        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        total_time_spent_in_actual_inference += duration_us;

        if (i > 0) {
            total_time_spent_in_actual_inference_excluding_first_image += duration_us;
            max_time_spent_in_actual_inference_per_image_excluding_first_image = std::max(
                max_time_spent_in_actual_inference_per_image_excluding_first_image, duration_us
            );
        }

        for (const auto& labeled_points : sample.labeled_points_by_class) {
            const uint16_t ground_truth_value = labeled_points.first;
            for (const dlib::point& point : labeled_points.second) {
                const uint16_t predicted_value = result_image.label_image(point.y(), point.x());
                ++confusion_matrix_per_pixel[ground_truth_value][predicted_value];
            }
            ground_truth_count += labeled_points.second.size();
        }

        update_confusion_matrix_per_region(confusion_matrix_per_region, sample.labeled_points_by_class, sample.label_image, result_image.label_image, update_confusion_matrix_per_region_temp);

        result_image_write_requests.enqueue(result_image);
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "\nAll " << files.size() << " images processed in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0 << " seconds!"
        << " (actual inference: " << total_time_spent_in_actual_inference.count() / 1000000.0 << " seconds)" << std::endl;

    if (files.size() > 1) {
        std::cout
            << "Processing time excluding the first image: "
            << "average = " << total_time_spent_in_actual_inference_excluding_first_image.count() / 1000.0 / (files.size() - 1) << " ms, "
            << "max = " << max_time_spent_in_actual_inference_per_image_excluding_first_image.count() / 1000.0 << " ms" << std::endl;
    }

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
        std::cout << std::endl << "Confusion matrix per pixel:" << std::endl;
        print_confusion_matrix(confusion_matrix_per_pixel, anno_classes);

        std::cout << std::endl << "Confusion matrix per region (two-way):" << std::endl;
        print_confusion_matrix(confusion_matrix_per_region, anno_classes);
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}
