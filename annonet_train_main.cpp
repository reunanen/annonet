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
#include "annonet_train.h"

#include "cpp-read-file-in-memory/read-file-in-memory.h"
#include "cxxopts/include/cxxopts.hpp"
#include "lru-timday/shared_lru_cache_using_std.h"
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>

#include <iostream>
#include <iterator>
#include <thread>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

rectangle make_cropping_rect_around_defect(
    int dim,
    point center
)
{
    return centered_rect(center, dim, dim);
}

// ----------------------------------------------------------------------------------------

namespace std {
    template <>
    struct hash<image_filenames> {
        std::size_t operator()(const image_filenames& image_filenames) const {
            return hash<string>()(image_filenames.image_filename + ", " + image_filenames.label_filename);
        }
    };

    bool operator ==(const image_filenames& a, const image_filenames& b) {
        return a.image_filename == b.image_filename
            && a.label_filename == b.label_filename;
    }
}

// ----------------------------------------------------------------------------------------

struct crop
{
    NetPimpl::input_type input_image;
    NetPimpl::training_label_type labels;

    // prevent having to re-allocate memory constantly
    dlib::matrix<uint16_t> temporary_unweighted_label_image;

    std::string warning;
    std::string error;
};

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
void add_random_noise(NetPimpl::input_type& image, double noise_level, dlib::rand& rnd)
{
    const long nr = image.nr();
    const long nc = image.nc();

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            int old_value = image(r, c);
            int noise = static_cast<int>(std::round(rnd.get_random_gaussian() * noise_level));
            int new_value = old_value + noise;
            int new_value_clamped = std::max(0, std::min(new_value, static_cast<int>(std::numeric_limits<uint8_t>::max())));
            image(r, c) = new_value_clamped;
        }
    }
}
#endif // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT

#if 0
struct randomly_crop_image_temp {
    NetPimpl::input_type input_image;
    dlib::matrix<uint16_t> label_image;
};

void randomly_crop_image(
    int dim,
    const sample& full_sample,
    crop& crop,
    dlib::rand& rnd,
    const cxxopts::Options& options,
    randomly_crop_image_temp& temp
)
{
    DLIB_CASSERT(!full_sample.labeled_points_by_class.empty());

    const size_t class_index = rnd.get_random_32bit_number() % full_sample.labeled_points_by_class.size();

    auto i = full_sample.labeled_points_by_class.begin();

    for (size_t j = 0; j < class_index; ++i, ++j) {
        DLIB_CASSERT(i != full_sample.labeled_points_by_class.end());
    }
    DLIB_CASSERT(i != full_sample.labeled_points_by_class.end());
    DLIB_CASSERT(!i->second.empty());

    const size_t point_index = rnd.get_random_64bit_number() % i->second.size();

    const double further_downscaling_factor = options["further-downscaling-factor"].as<double>();
    const int dim_before_downscaling = std::round(dim * further_downscaling_factor);

    const rectangle rect = random_rect_containing_point(rnd, i->second[point_index], dim_before_downscaling, dim_before_downscaling, dlib::rectangle(0, 0, full_sample.input_image.nc() - 1, full_sample.input_image.nr() - 1));

    const chip_details chip_details(rect, chip_dims(dim_before_downscaling, dim_before_downscaling));

    if (further_downscaling_factor > 1.0) {
        extract_image_chip(full_sample.input_image, chip_details, temp.input_image, interpolate_bilinear());
        extract_image_chip(full_sample.label_image, chip_details, temp.label_image, interpolate_nearest_neighbor());

        crop.input_image.set_size(dim, dim);
        crop.temporary_unweighted_label_image.set_size(dim, dim);

        dlib::resize_image(temp.input_image, crop.input_image, interpolate_bilinear());
        dlib::resize_image(temp.label_image, crop.temporary_unweighted_label_image, interpolate_nearest_neighbor());
    }
    else {
        extract_image_chip(full_sample.input_image, chip_details, crop.input_image, interpolate_bilinear());
        extract_image_chip(full_sample.label_image, chip_details, crop.temporary_unweighted_label_image, interpolate_nearest_neighbor());
    }

    // Randomly flip the input image and the labels.
    const bool allow_flip_left_right = options.count("allow-flip-left-right") > 0;
    const bool allow_flip_upside_down = options.count("allow-flip-upside-down") > 0;
    if (allow_flip_left_right && rnd.get_random_double() > 0.5) {
        crop.input_image = fliplr(crop.input_image);
        crop.label_image = fliplr(crop.label_image);
    }
    if (allow_flip_upside_down && rnd.get_random_double() > 0.5) {
        crop.input_image = flipud(crop.input_image);
        crop.label_image = flipud(crop.label_image);
    }

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
    double grayscale_noise_level_stddev = options["grayscale-noise-level-stddev"].as<double>();
    if (grayscale_noise_level_stddev > 0.0) {
        double grayscale_noise_level = fabs(rnd.get_random_gaussian() * grayscale_noise_level_stddev);
        add_random_noise(crop.input_image, grayscale_noise_level, rnd);
    }
#else // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
    const bool allow_random_color_offset = options.count("allow-random-color-offset") > 0;
    if (allow_random_color_offset) {
        apply_random_color_offset(crop.input_image, rnd);
    }
#endif // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
}
#endif

// ----------------------------------------------------------------------------------------

std::string read_anno_classes_file(const std::string& folder)
{
    const std::vector<file> files = get_files_in_directory_tree(folder,
        [](const file& name) {
            return name.name() == "anno_classes.json";
        }, 0); // do not scan subdirectories - the file must be in the root

    if (files.empty()) {
        std::cout << "Warning: no anno_classes.json file found in " + folder << std::endl;
        std::cout << " --> Using the default anno classes" << std::endl;
        return "";
    }

    if (files.size() > 1) {
        throw std::runtime_error("Found multiple anno_classes.json files - this shouldn't happen");
    }

    const std::string json = read_file_as_string(files.front());

    return json;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        cout << "To run this program you need data annotated using the anno program." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./annonet_train /path/to/anno/data" << endl;
        return 1;
    }

    cxxopts::Options options("annonet_train", "Train semantic-segmentation networks using data generated in anno");

    std::ostringstream default_data_loader_thread_count;
    default_data_loader_thread_count << std::thread::hardware_concurrency();

    options.add_options()
        ("d,downscaling-factor", "The downscaling factor (>= 1.0)", cxxopts::value<double>()->default_value("1.0"))
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
#if 0
        ("u,allow-flip-upside-down", "Randomly flip input images upside down")
        ("l,allow-flip-left-right", "Randomly flip input images horizontally")
#endif
#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
        ("n,grayscale-noise-level-stddev", "Set the standard deviation of the level of grayscale noise to add", cxxopts::value<double>()->default_value("0.0"))
#else // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
        ("o,allow-random-color-offset", "Randomly apply color offsets")
#endif // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
#if 0
        ("ignore-class", "Ignore specific classes by index", cxxopts::value<std::vector<uint16_t>>())
#endif
        ("b,minibatch-size", "Set minibatch size", cxxopts::value<size_t>()->default_value("100"))
        ("net-width-scaler", "Scaler of net width", cxxopts::value<double>()->default_value("1.0"))
        ("net-width-min-filter-count", "Minimum net width filter count", cxxopts::value<int>()->default_value("1"))
        ("initial-learning-rate", "Set initial learning rate", cxxopts::value<double>()->default_value("0.1"))
        ("learning-rate-shrink-factor", "Set learning rate shrink factor", cxxopts::value<double>()->default_value("0.1"))
        ("min-learning-rate", "Set minimum learning rate", cxxopts::value<double>()->default_value("1e-6"))
        ("save-interval", "Save the resulting inference network every this many steps", cxxopts::value<size_t>()->default_value("1000"))
        ("t,relative-training-length", "Relative training length", cxxopts::value<double>()->default_value("2.0"))
        ("c,cached-image-count", "Cached image count", cxxopts::value<int>()->default_value("8"))
        ("data-loader-thread-count", "Number of data loader threads", cxxopts::value<unsigned int>()->default_value(default_data_loader_thread_count.str()))
        ("no-empty-label-image-warning", "Do not warn about empty label images")
        ;

    try {
        options.parse_positional("input-directory");
        options.parse(argc, argv);

        cxxopts::check_required(options, { "input-directory" });

        std::cout << "Input directory = " << options["input-directory"].as<std::string>() << std::endl;
        std::cout << "Downscaling factor = " << options["downscaling-factor"].as<double>() << std::endl;

        if (options["downscaling-factor"].as<double>() <= 0.0) {
            throw std::runtime_error("The downscaling factor has to be strictly positive.");
        }
    }
    catch (std::exception& e) {
        cerr << e.what() << std::endl;
        cerr << std::endl;
        cerr << options.help() << std::endl;
        return 2;
    }

    const double downscaling_factor = options["downscaling-factor"].as<double>();
#if 0
    const bool allow_flip_upside_down = options.count("allow-flip-upside-down") > 0;
#endif
#if 0
    const std::vector<uint16_t> classes_to_ignore = options["ignore-class"].as<std::vector<uint16_t>>();
#endif
    const auto minibatch_size = options["minibatch-size"].as<size_t>();
    const auto net_width_scaler = options["net-width-scaler"].as<double>();
    const auto net_width_min_filter_count = options["net-width-min-filter-count"].as<int>();
    const auto initial_learning_rate = options["initial-learning-rate"].as<double>();
    const auto learning_rate_shrink_factor = options["learning-rate-shrink-factor"].as<double>();
    const auto min_learning_rate = options["min-learning-rate"].as<double>();
    const auto save_interval = options["save-interval"].as<size_t>();
    const auto relative_training_length = std::max(0.01, options["relative-training-length"].as<double>());
    const auto cached_image_count = options["cached-image-count"].as<int>();
    const auto data_loader_thread_count = std::max(1U, options["data-loader-thread-count"].as<unsigned int>());
    const bool warn_about_empty_label_images = options.count("no-empty-label-image-warning") == 0;

#if 0
    std::cout << "Allow flipping input images upside down = " << (allow_flip_upside_down ? "yes" : "no") << std::endl;
#endif
    std::cout << "Minibatch size = " << minibatch_size << std::endl;
    std::cout << "Net width scaler = " << net_width_scaler << ", min filter count = " << net_width_min_filter_count << std::endl;
    std::cout << "Initial learning rate = " << initial_learning_rate << std::endl;
    std::cout << "Learning rate shrink factor = " << learning_rate_shrink_factor << std::endl;
    std::cout << "Min learning rate = " << min_learning_rate << std::endl;
    std::cout << "Save interval = " << save_interval << std::endl;
    std::cout << "Relative training length = " << relative_training_length << std::endl;
    std::cout << "Cached image count = " << cached_image_count << std::endl;
    std::cout << "Data loader thread count = " << data_loader_thread_count << std::endl;

#if 0
    if (!classes_to_ignore.empty()) {
        std::cout << "Classes to ignore =";
        for (uint16_t class_to_ignore : classes_to_ignore) {
            std::cout << " " << class_to_ignore;
        }
        std::cout << std::endl;
    }
#endif

    const auto anno_classes_json = read_anno_classes_file(options["input-directory"].as<std::string>());
    const auto anno_classes = parse_anno_classes(anno_classes_json);

    if (anno_classes.size() > 0 && anno_classes[0].classlabel != "<<ignore>>") {
        cout << "WARNING: the label of the first class is \'" << anno_classes[0].classlabel << "', and not '<<ignore>>' as expected" << std::endl;
    }

    const unsigned long iterations_without_progress_threshold = static_cast<unsigned long>(std::round(relative_training_length * 2000));
    const unsigned long previous_loss_values_dump_amount = static_cast<unsigned long>(std::round(relative_training_length * 400));
    const unsigned long batch_normalization_running_stats_window_size = static_cast<unsigned long>(std::round(relative_training_length * 100));

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto image_files = find_image_files(options["input-directory"].as<std::string>(), true);
    cout << "images in dataset: " << image_files.size() << endl;
    if (image_files.size() == 0)
    {
        cout << "Didn't find an anno dataset. " << endl;
        return 1;
    }

    cout << "\nProcessing labels...\n" << endl;

    std::vector<std::vector<dlib::mmod_rect>> all_labels;
    all_labels.reserve(image_files.size());

    for (const auto& image_filenames : image_files) {
        const std::string json = read_file_as_string(image_filenames.label_filename);
        all_labels.push_back(parse_labels(json, anno_classes));        
    }

    dlib::mmod_options mmod_options(all_labels, 40, 40);

    std::cout << "Detector windows:" << std::endl;
    for (const auto& detector_window : mmod_options.detector_windows) {
        std::cout << " - " << detector_window.label << ": " << detector_window.width << " x " << detector_window.height << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Overlap NMS IOU threshold:             " << mmod_options.overlaps_nms.get_iou_thresh() << std::endl;
    std::cout << "Overlap NMS percent covered threshold: " << mmod_options.overlaps_nms.get_percent_covered_thresh() << std::endl;

    NetPimpl::TrainingNet training_net;

    training_net.Initialize(mmod_options, NetPimpl::GetDefaultSolver(), net_width_scaler, net_width_min_filter_count);
    training_net.SetSynchronizationFile("annonet_trainer_state_file.dat", std::chrono::seconds(10 * 60));
    training_net.BeVerbose();
    training_net.SetLearningRate(initial_learning_rate);
    training_net.SetLearningRateShrinkFactor(learning_rate_shrink_factor);
    training_net.SetIterationsWithoutProgressThreshold(iterations_without_progress_threshold);
    training_net.SetPreviousLossValuesDumpAmount(previous_loss_values_dump_amount);
    training_net.SetAllBatchNormalizationRunningStatsWindowSizes(batch_normalization_running_stats_window_size);

#if 0
    const auto ignore_classes_to_ignore = [&classes_to_ignore](sample& sample) {
        for (const auto class_to_ignore : classes_to_ignore) {
            const auto i = sample.labeled_points_by_class.find(class_to_ignore);
            if (i != sample.labeled_points_by_class.end()) {
                for (const dlib::point& point : i->second) {
                    sample.label_image(point.y(), point.x()) = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
                }
                sample.labeled_points_by_class.erase(class_to_ignore);
            }
        }
    };
#endif

    shared_lru_cache_using_std<image_filenames, std::shared_ptr<sample>, std::unordered_map> full_images_cache(
        [&](const image_filenames& image_filenames) {
            std::shared_ptr<sample> sample(new sample);
            *sample = read_sample(image_filenames, anno_classes, true, downscaling_factor);
#if 0
            ignore_classes_to_ignore(*sample);
#endif
            return sample;
        }, cached_image_count);

    cout << endl << "Now training..." << endl;
   
    set_low_priority();

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<crop> data(2 * minibatch_size);
    auto pull_crops = [&data, &full_images_cache, &image_files, &options](time_t seed)
    {
        const auto timed_seed = time(0) + seed;

        dlib::random_cropper cropper;
        cropper.set_seed(timed_seed);
        cropper.set_chip_dims(200, 200);
        cropper.set_min_object_size(40, 40);
        cropper.set_max_rotation_degrees(10);

        dlib::rand rnd(timed_seed);

        const bool allow_random_color_offset = options.count("allow-random-color-offset") > 0;

        NetPimpl::input_type input_image;
        crop crop;

        std::vector<NetPimpl::input_type> cropped_input_image;
        std::vector<NetPimpl::training_label_type> cropped_labels;

        while (data.is_enabled())
        {
            crop.error.clear();
            crop.warning.clear();

            const size_t index = rnd.get_random_32bit_number() % image_files.size();
            const image_filenames& image_filenames = image_files[index];
            const std::shared_ptr<sample> ground_truth_sample = full_images_cache(image_filenames);

            const std::vector<NetPimpl::input_type> images = { ground_truth_sample->input_image };
            const std::vector<std::vector<dlib::mmod_rect>> labels = { ground_truth_sample->labels };

            if (!ground_truth_sample->error.empty()) {
                crop.error = ground_truth_sample->error;
            }
            else if (ground_truth_sample->labels.empty()) {
                crop.warning = "Warning: no annotation paths in " + ground_truth_sample->image_filenames.label_filename;
            }
            else {
                cropper(1, images, labels, cropped_input_image, cropped_labels);
                if (cropped_input_image.size() != 1 || cropped_labels.size() != 1) {
                    crop.warning = "Warning: unexpected cropping result";
                }
                else {
                    crop.input_image = cropped_input_image.front();
                    crop.labels = cropped_labels.front();

                    if (allow_random_color_offset) {
                        apply_random_color_offset(crop.input_image, rnd);
                    }
                }
            }
            data.enqueue(crop);
        }
    };

    std::vector<std::thread> data_loaders;
    for (unsigned int i = 0; i < data_loader_thread_count; ++i) {
        data_loaders.push_back(std::thread([pull_crops, i]() { pull_crops(i); }));
    }
    
    size_t minibatch = 0;

    const auto save_inference_net = [&]() {
        const NetPimpl::RuntimeNet runtime_net = training_net.GetRuntimeNet();
        
        std::ostringstream serialized;
        runtime_net.Serialize(serialized);

        cout << "saving network" << endl;
        serialize("annonet.dnn") << anno_classes_json << downscaling_factor << serialized.str();
    };

    std::vector<NetPimpl::input_type> samples;
    std::vector<NetPimpl::training_label_type> labels;

    std::set<std::string> warnings_already_printed;

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    while (training_net.GetLearningRate() >= min_learning_rate)
    {
        samples.clear();
        labels.clear();

        // make a mini-batch
        crop crop;
        while (samples.size() < minibatch_size)
        {
            data.dequeue(crop);

            if (!crop.error.empty()) {
                throw std::runtime_error(crop.error);
            }
            else if (!crop.warning.empty()) {
                if (warn_about_empty_label_images && warnings_already_printed.find(crop.warning) == warnings_already_printed.end()) {
                    std::cout << crop.warning << std::endl;
                    warnings_already_printed.insert(crop.warning);
                }
            }
            else {
                samples.push_back(std::move(crop.input_image));
                labels.push_back(std::move(crop.labels));
            }
        }

        training_net.StartTraining(samples, labels);

        if (minibatch++ % save_interval == 0) {
            save_inference_net();
        }
    }

    // Training done: tell threads to stop.
    data.disable();

    const auto join = [](std::vector<thread>& threads)
    {
        for (std::thread& thread : threads) {
            thread.join();
        }
    };

    join(data_loaders);

    save_inference_net();
}
catch(std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}