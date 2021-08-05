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
*/

#include "annonet.h"
#include "annonet_infer.h"

#include "cxxopts/include/cxxopts.hpp"
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_saver/save_png.h>

#include <rapidjson/prettywriter.h>

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

#if 0
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

struct update_confusion_matrix_per_region_temp
{
    dlib::matrix<int> ground_truth_blobs;
    dlib::matrix<int> result_blobs;
};

void update_confusion_matrix_per_region(
    confusion_matrix_type& confusion_matrix_per_region,
    const std::unordered_map<uint16_t, std::deque<dlib::point>>& labeled_points_by_class,
    const dlib::matrix<uint16_t>& ground_truth_label_image,
    const dlib::matrix<uint16_t>& result_label_image,
    update_confusion_matrix_per_region_temp& temp = update_confusion_matrix_per_region_temp()
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
#endif

// ----------------------------------------------------------------------------------------

struct result_image_type {
    std::string filename;
    int original_width = 0;
    int original_height = 0;
    std::vector<dlib::mmod_rect> labels;
};

inline uint16_t classlabel_to_index_label(const std::string& classlabel, const std::vector<AnnoClass>& anno_classes)
{
    for (const AnnoClass& anno_class : anno_classes) {
        if (anno_class.classlabel == classlabel) {
            return anno_class.index;
        }
    }
    throw std::runtime_error("Unknown class: '" + classlabel + "'");
}

void write_labels(const std::string& filename, const std::vector<dlib::mmod_rect>& labels, const std::vector<AnnoClass>& anno_classes)
{
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

    writer.StartArray();

    for (const auto& label : labels) {

        const auto& index = classlabel_to_index_label(label.label, anno_classes);
        const auto& anno_class = anno_classes[index];

        writer.StartObject();
        writer.String("color");
        writer.StartObject();
        {
            writer.String("r"); writer.Int(anno_class.rgba_label.red);
            writer.String("g"); writer.Int(anno_class.rgba_label.green);
            writer.String("b"); writer.Int(anno_class.rgba_label.blue);
            writer.String("a"); writer.Int(anno_class.rgba_label.alpha);
        }
        writer.EndObject();

        writer.String("color_paths");
        writer.StartArray();

        {
            writer.StartArray();
            {
                writer.StartObject();
                writer.String("x"); writer.Int(label.rect.left());
                writer.String("y"); writer.Int(label.rect.top());
                writer.EndObject();
            }
            {
                writer.StartObject();
                writer.String("x"); writer.Int(label.rect.right());
                writer.String("y"); writer.Int(label.rect.top());
                writer.EndObject();
            }
            {
                writer.StartObject();
                writer.String("x"); writer.Int(label.rect.right());
                writer.String("y"); writer.Int(label.rect.bottom());
                writer.EndObject();
            }
            {
                writer.StartObject();
                writer.String("x"); writer.Int(label.rect.left());
                writer.String("y"); writer.Int(label.rect.bottom());
                writer.EndObject();
            }
            writer.EndArray();
        }

        writer.EndArray();

        writer.String("detection_confidence");
        writer.Double(label.detection_confidence);

        writer.EndObject();
    }

    writer.EndArray();

    std::ofstream out(filename);
    out << buffer.GetString();
}

std::vector<double> convert_gains_by_class_to_gains_by_detector_window(const std::vector<double>& gains_by_class, const std::vector<AnnoClass>& anno_classes, const dlib::mmod_options& mmod_options)
{
    DLIB_CASSERT(gains_by_class.size() == anno_classes.size());

    std::vector<double> gains_by_detector_window(mmod_options.detector_windows.size());

    for (size_t detector_window_index = 0, end = gains_by_detector_window.size(); detector_window_index < end; ++detector_window_index) {
        const std::string& classlabel = mmod_options.detector_windows[detector_window_index].label;
        const auto classlabel_index = classlabel_to_index_label(classlabel, anno_classes);
        gains_by_detector_window[detector_window_index] = gains_by_class[classlabel_index];
    }
    
    return gains_by_detector_window;
}

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
    const std::string default_max_tile_width = "2048";
    const std::string default_max_tile_height = "2048";
#else
    // in CPU-only mode, we can handle larger tiles
    const std::string default_max_tile_width = "4096";
    const std::string default_max_tile_height = "4096";
#endif

    options.add_options()
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
        ("g,gain", "Supply a class-specific gain, for example: 1:-0.5", cxxopts::value<std::vector<std::string>>())
        ("w,tile-max-width", "Set max tile width", cxxopts::value<int>()->default_value(default_max_tile_width))
        ("h,tile-max-height", "Set max tile height", cxxopts::value<int>()->default_value(default_max_tile_height))
        ("limit-to-size", "Limit tiles to image size")
        ("full-image-reader-thread-count", "Set the number of full-image reader threads", cxxopts::value<int>()->default_value(hardware_concurrency.str()))
        ("result-image-writer-thread-count", "Set the number of result-image writer threads", cxxopts::value<int>()->default_value(hardware_concurrency.str()))
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

    DLIB_CASSERT(anno_classes.size() >= 2);

    const auto get_gains_by_detector_window = [&options, &anno_classes, &net]() {
        const std::vector<double> gains_by_class = parse_class_specific_values(options["gain"].as<std::vector<std::string>>(), anno_classes.size());

        assert(gains_by_class.size() == anno_classes.size());

        std::cout << "Using gains:" << std::endl;
        for (size_t class_index = 0, end = gains_by_class.size(); class_index < end; ++class_index) {
            const auto& classlabel = anno_classes[class_index].classlabel;
            if (classlabel != "<<ignore>>") {
                std::cout << " - " << class_index << " (" << classlabel << "): " << gains_by_class[class_index] << std::endl;
            }
        }
        std::cout << std::endl;

        return convert_gains_by_class_to_gains_by_detector_window(gains_by_class, anno_classes, net.GetOptions());
    };

    const std::vector<double> gains_by_detector_window = get_gains_by_detector_window();

    set_low_priority();

    annonet_infer_temp temp;
    matrix<uint16_t> index_label_tile_resized;

    auto files = find_image_files(options["input-directory"].as<std::string>(), false);

    dlib::pipe<image_filenames> full_image_read_requests(files.size());
    for (const image_filenames& file : files) {
        full_image_read_requests.enqueue(image_filenames(file));
    }

    const int full_image_reader_count = std::max(1, options["full-image-reader-thread-count"].as<int>());
    const int result_image_writer_count = std::max(1, options["result-image-writer-thread-count"].as<int>());

    dlib::pipe<sample> full_image_read_results(full_image_reader_count);

    std::vector<std::thread> full_image_readers;

    for (unsigned int i = 0; i < full_image_reader_count; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            image_filenames image_filenames;
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
            while (result_image_write_requests.dequeue(result_image)) {
                if (downscaling_factor != 1.0) {
                    for (auto& label : result_image.labels) {
                        const auto scale = [downscaling_factor](const long value) {
                            return static_cast<long>(std::round(value * downscaling_factor));
                        };
                        label.rect.set_left  (scale(label.rect.left  ()));
                        label.rect.set_right (scale(label.rect.right ()));
                        label.rect.set_top   (scale(label.rect.top   ()));
                        label.rect.set_bottom(scale(label.rect.bottom()));
                    }
                }
                write_labels(result_image.filename, result_image.labels, anno_classes);
                result_image_write_results.enqueue(true);
            }
        }));
    }

#if 0
    const int min_input_dimension = NetPimpl::TrainingNet::GetRequiredInputDimension();
#else
    const int min_input_dimension = 16;
#endif

    tiling::parameters tiling_parameters;
    tiling_parameters.max_tile_width = options["tile-max-width"].as<int>();
    tiling_parameters.max_tile_height = options["tile-max-height"].as<int>();
    tiling_parameters.overlap_x = min_input_dimension;
    tiling_parameters.overlap_y = min_input_dimension;
    tiling_parameters.limit_to_size = options.count("limit-to-size") > 0;

    DLIB_CASSERT(tiling_parameters.max_tile_width >= min_input_dimension);
    DLIB_CASSERT(tiling_parameters.max_tile_height >= min_input_dimension);

#if 0
    // first index: ground truth, second index: predicted
    confusion_matrix_type confusion_matrix_per_pixel, confusion_matrix_per_region;
    init_confusion_matrix(confusion_matrix_per_pixel, anno_classes.size());
    init_confusion_matrix(confusion_matrix_per_region, anno_classes.size());
    size_t ground_truth_count = 0;
#endif

    const auto t0 = std::chrono::steady_clock::now();

#if 0
    update_confusion_matrix_per_region_temp update_confusion_matrix_per_region_temp;
#endif

    std::unordered_map<std::string, size_t> hit_counts;

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

        result_image.filename = sample.image_filenames.image_filename + "_result_path.json";
        result_image.original_width = sample.original_width;
        result_image.original_height = sample.original_height;

        annonet_infer(net, sample.input_image, result_image.labels, gains_by_detector_window, tiling_parameters, temp);

#if 0
        for (const auto& labeled_points : sample.labeled_points_by_class) {
            const uint16_t ground_truth_value = labeled_points.first;
            for (const dlib::point& point : labeled_points.second) {
                const uint16_t predicted_value = result_image.label_image(point.y(), point.x());
                ++confusion_matrix_per_pixel[ground_truth_value][predicted_value];
            }
            ground_truth_count += labeled_points.second.size();
        }

        update_confusion_matrix_per_region(confusion_matrix_per_region, sample.labeled_points_by_class, sample.label_image, result_image.label_image, update_confusion_matrix_per_region_temp);
#endif

        for (const auto& label : result_image.labels) {
            ++hit_counts[label.label];
        }

        result_image_write_requests.enqueue(result_image);
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "\nAll " << files.size() << " images processed in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0 << " seconds!" << std::endl;

    std::cout << std::endl << "Hit counts:" << std::endl;
    for (size_t i = 0, end = anno_classes.size(); i < end; ++i) {
        const auto& classlabel = anno_classes[i].classlabel;
        if (classlabel != "<<ignore>>") {
            std::cout << " - " << i << " (" << classlabel << "): " << hit_counts[classlabel] << std::endl;
        }
    }
    std::cout << std::endl;

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

#if 0
    if (ground_truth_count) {
        std::cout << std::endl << "Confusion matrix per pixel:" << std::endl;
        print_confusion_matrix(confusion_matrix_per_pixel, anno_classes);

        std::cout << std::endl << "Confusion matrix per region (two-way):" << std::endl;
        print_confusion_matrix(confusion_matrix_per_region, anno_classes);
    }
#endif
}
catch(std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}