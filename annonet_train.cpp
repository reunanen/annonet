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

#include "cpp-read-file-in-memory/read-file-in-memory.h"
#include "cxxopts/include/cxxopts.hpp"
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>

#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>
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

struct crop
{
    NetPimpl::input_type input_image;
    NetPimpl::training_label_type label_image;

    // prevent having to re-allocate memory constantly
    dlib::matrix<uint16_t> temporary_unweighted_label_image;
};

void find_equal_class_weights (
    const dlib::matrix<uint16_t>& unweighted_label_image,
    NetPimpl::training_label_type& weighted_label_image
)
{
    const long nr = unweighted_label_image.nr();
    const long nc = unweighted_label_image.nc();

    std::unordered_map<uint16_t, size_t> label_counts;

    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            const uint16_t label = unweighted_label_image(r, c);
            if (label != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                ++label_counts[label];
            }
        }
    }

    const size_t total_count = std::accumulate(label_counts.begin(), label_counts.end(), 0,
        [&](size_t total, const std::pair<uint16_t, size_t>& item) { return total + item.second; });

    DLIB_CASSERT(total_count > 0);

    const double average_weight = nr * nc / static_cast<double>(total_count);
    const double average_count = total_count / static_cast<double>(label_counts.size());

    std::unordered_map<uint16_t, double> label_weights;

    // try 0.0 for equally balanced pixels, and 1.0 for equally balanced classes
    constexpr double a_priori_weight = 0.0;

    for (const auto& item : label_counts) {
        label_weights[item.first] = average_weight * pow(average_count / item.second, a_priori_weight);
    }

    weighted_label_image.set_size(nr, nc);

    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            const uint16_t label = unweighted_label_image(r, c);
            const double weight = label == dlib::loss_multiclass_log_per_pixel_::label_to_ignore ? 0.0 : label_weights[label];
            weighted_label_image(r, c) = dlib::loss_multiclass_log_per_pixel_weighted_::weighted_label(label, weight);
        }
    }
}

void randomly_crop_image(
    int dim,
    const sample& full_sample,
    crop& crop,
    dlib::rand& rnd,
    bool allow_flip_upside_down
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

    const rectangle rect = centered_rect(i->second[point_index], dim, dim);

    const chip_details chip_details(rect, chip_dims(dim, dim));

    // Crop the input image.
    extract_image_chip(full_sample.input_image, chip_details, crop.input_image, interpolate_bilinear());

    // Crop the labels correspondingly. However, note that here bilinear
    // interpolation would make absolutely no sense.
    // TODO: mark all invalid areas as ignore.
    extract_image_chip(full_sample.label_image, chip_details, crop.temporary_unweighted_label_image, interpolate_nearest_neighbor());

    find_equal_class_weights(crop.temporary_unweighted_label_image, crop.label_image);

    // Also randomly flip the input image and the labels.
    if (allow_flip_upside_down && rnd.get_random_double() > 0.5) {
        crop.input_image = flipud(crop.input_image);
        crop.label_image = flipud(crop.label_image);
    }
}

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

    options.add_options()
        ("d,downscaling-factor", "The downscaling factor (>= 1.0)", cxxopts::value<double>()->default_value("1.0"))
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
        ("u,allow-flip-upside-down", "Randomly flip input images upside down")
        ("ignore-class", "Ignore specific classes by index", cxxopts::value<std::vector<uint16_t>>())
        ("b,minibatch-size", "Set minibatch size", cxxopts::value<size_t>()->default_value("100"))
        ("save-interval", "Save the resulting inference network every this many steps", cxxopts::value<size_t>()->default_value("1000"))
        ("t,relative-training-length", "Relative training length", cxxopts::value<double>()->default_value("2.0"))
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
    const bool allow_flip_upside_down = options.count("allow-flip-upside-down") > 0;
    const std::vector<uint16_t> classes_to_ignore = options["ignore-class"].as<std::vector<uint16_t>>();
    const auto minibatch_size = options["minibatch-size"].as<size_t>();
    const auto save_interval = options["save-interval"].as<size_t>();
    const auto relative_training_length = std::max(0.01, options["relative-training-length"].as<double>());

    std::cout << "Allow flipping input images upside down = " << (allow_flip_upside_down ? "yes" : "no") << std::endl;
    std::cout << "Minibatch size = " << minibatch_size << std::endl;
    std::cout << "Save interval = " << save_interval << std::endl;
    std::cout << "Relative training length = " << relative_training_length << std::endl;

    if (!classes_to_ignore.empty()) {
        std::cout << "Classes to ignore =";
        for (uint16_t class_to_ignore : classes_to_ignore) {
            std::cout << " " << class_to_ignore;
        }
        std::cout << std::endl;
    }

    const int required_input_dimension = NetPimpl::TrainingNet::GetRequiredInputDimension();
    std::cout << "Required input dimension = " << required_input_dimension << std::endl;

    const auto anno_classes_json = read_anno_classes_file(options["input-directory"].as<std::string>());
    const auto anno_classes = parse_anno_classes(anno_classes_json);

    const double initial_learning_rate = 0.1;
    const double learning_rate_shrink_factor = 0.1;
    const double min_learning_rate = 1e-6;
    const unsigned long iterations_without_progress_threshold = static_cast<unsigned long>(std::round(relative_training_length * 2000));
    const unsigned long previous_loss_values_dump_amount = static_cast<unsigned long>(std::round(relative_training_length * 400));
    const unsigned long batch_normalization_running_stats_window_size = static_cast<unsigned long>(std::round(relative_training_length * 100));

    NetPimpl::TrainingNet training_net;

    std::vector<NetPimpl::input_type> samples;
    std::vector<NetPimpl::training_label_type> labels;

    { // Test that the input size is correct for the net that we have built, and also that a minibatch fits in GPU memory
        training_net.Initialize();
        training_net.SetClassCount(minibatch_size);

        for (uint16_t label = 0; label < minibatch_size; ++label) {
            NetPimpl::input_type input_image(required_input_dimension, required_input_dimension);
            NetPimpl::training_label_type label_image(required_input_dimension, required_input_dimension);
            //input_image = label * 255 / (minibatch_size - 1);
            label_image = label;

            samples.push_back(std::move(input_image));
            labels.push_back(std::move(label_image));
        }

        training_net.StartTraining(samples, labels);
    }

    training_net.Initialize();
    training_net.SetClassCount(anno_classes.size());
    training_net.SetLearningRate(initial_learning_rate);
    training_net.SetLearningRateShrinkFactor(learning_rate_shrink_factor);
    training_net.SetIterationsWithoutProgressThreshold(iterations_without_progress_threshold);
    training_net.SetPreviousLossValuesDumpAmount(previous_loss_values_dump_amount);
    training_net.SetAllBatchNormalizationRunningStatsWindowSizes(batch_normalization_running_stats_window_size);
    training_net.SetSynchronizationFile("annonet_trainer_state_file.dat", std::chrono::seconds(10 * 60));
    training_net.BeVerbose();

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto image_files = find_image_files(options["input-directory"].as<std::string>(), true);
    cout << "images in dataset: " << image_files.size() << endl;
    if (image_files.size() == 0)
    {
        cout << "Didn't find an anno dataset. " << endl;
        return 1;
    }

    dlib::pipe<image_filenames> full_image_read_requests(image_files.size());

    for (const image_filenames& original : image_files) {
        image_filenames copy = original;
        full_image_read_requests.enqueue(copy);
    }

    dlib::pipe<sample> full_image_read_results(std::thread::hardware_concurrency());

    std::vector<std::thread> full_image_readers;
    
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

    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            image_filenames image_filenames;
            while (full_image_read_requests.dequeue(image_filenames)) {
                sample sample = read_sample(image_filenames, anno_classes, true, downscaling_factor);
                ignore_classes_to_ignore(sample);
                full_image_read_results.enqueue(sample);
            }
        }));
    }

    std::deque<sample> full_images;

    for (size_t i = 0, end = image_files.size(); i < end; ++i) {
        std::cout << "\rReading image " << (i + 1) << " of " << end << "...";
        sample ground_truth_sample;
        full_image_read_results.dequeue(ground_truth_sample);
        if (!ground_truth_sample.error.empty()) {
            throw std::runtime_error(ground_truth_sample.error);
        }
        if (!ground_truth_sample.labeled_points_by_class.empty()) {
            full_images.push_back(std::move(ground_truth_sample));
        }
        else {
            std::cout << std::endl << "Warning: no labeled points in " << ground_truth_sample.image_filenames.label_filename << std::endl;
        }
    }

    full_image_read_requests.disable();

    cout << endl << "Now training..." << endl;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<crop> data(2 * minibatch_size);
    auto pull_crops = [&data, &full_images, required_input_dimension, allow_flip_upside_down](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        NetPimpl::input_type input_image;
        matrix<uint16_t> index_label_image;
        crop crop;
        while (data.is_enabled())
        {
            const size_t index = rnd.get_random_32bit_number() % full_images.size();
            const sample& ground_truth_sample = full_images[index];
            randomly_crop_image(required_input_dimension, ground_truth_sample, crop, rnd, allow_flip_upside_down);
            data.enqueue(crop);
        }
    };

    std::vector<std::thread> data_loaders;
    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
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

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    while (training_net.GetLearningRate() >= min_learning_rate)
    {
        samples.clear();
        labels.clear();

        // make a mini-batch
        crop crop;
        while(samples.size() < minibatch_size)
        {
            data.dequeue(crop);

            samples.push_back(std::move(crop.input_image));
            labels.push_back(std::move(crop.label_image));
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

    join(full_image_readers);
    join(data_loaders);

    save_inference_net();
}
catch(std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}