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
            return hash<string>()(
                image_filenames.input0_filename + ", " +
                image_filenames.input1_filename + ", " +
                image_filenames.ground_truth_filename
            );
        }
    };

    bool operator ==(const image_filenames& a, const image_filenames& b) {
        return a.input0_filename == b.input0_filename
            && a.input1_filename == b.input1_filename
            && a.ground_truth_filename == b.ground_truth_filename;
    }
}

// ----------------------------------------------------------------------------------------

struct crop
{
    NetPimpl::input_type input_image_stack;
    NetPimpl::training_label_type target_image;

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

struct randomly_crop_image_temp {
    NetPimpl::input_type input_image_stack;
    NetPimpl::training_label_type target_image;
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
    // TODO: handle borders
    const size_t x = rnd.get_random_64bit_number() % full_sample.input_image_stack[0].nc();
    const size_t y = rnd.get_random_64bit_number() % full_sample.input_image_stack[0].nr();

    const double further_downscaling_factor = options["further-downscaling-factor"].as<double>();
    const int dim_before_downscaling = std::round(dim * further_downscaling_factor);

    const rectangle rect = random_rect_containing_point(rnd, dlib::point(x, y), dim_before_downscaling, dim_before_downscaling, dlib::rectangle(0, 0, full_sample.input_image_stack[0].nc() - 1, full_sample.input_image_stack[0].nr() - 1));

    const chip_details chip_details(rect, chip_dims(dim_before_downscaling, dim_before_downscaling));

    crop.input_image_stack.resize(full_sample.input_image_stack.size());

    if (further_downscaling_factor > 1.0) {
        temp.input_image_stack.resize(full_sample.input_image_stack.size());

        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            extract_image_chip(full_sample.input_image_stack[i], chip_details, temp.input_image_stack[i]);
        }
        extract_image_chip(full_sample.target_image, chip_details, temp.target_image);

        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            crop.input_image_stack[i].set_size(dim, dim);
        }
        crop.target_image.set_size(dim, dim);

        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            dlib::resize_image(temp.input_image_stack[i], crop.input_image_stack[i]);
        }
        dlib::resize_image(temp.target_image, crop.target_image);
    }
    else {
        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            extract_image_chip(full_sample.input_image_stack[i], chip_details, crop.input_image_stack[i]);
        }
        extract_image_chip(full_sample.target_image, chip_details, crop.target_image);
    }

    // Randomly flip the input image and the labels.
    const bool allow_flip_left_right = options.count("allow-flip-left-right") > 0;
    const bool allow_flip_upside_down = options.count("allow-flip-upside-down") > 0;
    if (allow_flip_left_right && rnd.get_random_double() > 0.5) {
        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            crop.input_image_stack[i] = fliplr(crop.input_image_stack[i]);
        }
        crop.target_image = fliplr(crop.target_image);
    }
    if (allow_flip_upside_down && rnd.get_random_double() > 0.5) {
        for (size_t i = 0, end = full_sample.input_image_stack.size(); i < end; ++i) {
            crop.input_image_stack[i] = flipud(crop.input_image_stack[i]);
        }
        crop.target_image = flipud(crop.target_image);
    }

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
    double grayscale_noise_level_stddev = options["grayscale-noise-level-stddev"].as<double>();
    if (grayscale_noise_level_stddev > 0.0) {
        // TODO think this through
        double grayscale_noise_level = fabs(rnd.get_random_gaussian() * grayscale_noise_level_stddev);
        add_random_noise(crop.input_image_stack[0], grayscale_noise_level, rnd);
        add_random_noise(crop.input_image_stack[1], grayscale_noise_level, rnd);
    }
#else // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
    const bool allow_random_color_offset = options.count("allow-random-color-offset") > 0;
    if (allow_random_color_offset) {
        // TODO think this through
        apply_random_color_offset(crop.input_image_stack[0], rnd);
        apply_random_color_offset(crop.input_image_stack[1], rnd);
    }
#endif // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
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
        ("d,initial-downscaling-factor", "The initial downscaling factor (>= 1.0)", cxxopts::value<double>()->default_value("1.0"))
        ("f,further-downscaling-factor", "The further downscaling factor (>= 1.0)", cxxopts::value<double>()->default_value("1.0"))
        ("i,input-directory", "Input image directory", cxxopts::value<std::string>())
        ("u,allow-flip-upside-down", "Randomly flip input images upside down")
        ("l,allow-flip-left-right", "Randomly flip input images horizontally")
#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
        ("n,grayscale-noise-level-stddev", "Set the standard deviation of the level of grayscale noise to add", cxxopts::value<double>()->default_value("0.0"))
#else // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
        ("o,allow-random-color-offset", "Randomly apply color offsets")
#endif // DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
        ("b,minibatch-size", "Set minibatch size", cxxopts::value<size_t>()->default_value("100"))
        ("input-dimension-multiplier", "Size of input patches, relative to minimum required", cxxopts::value<double>()->default_value("3.0"))
        ("net-width-scaler", "Scaler of net width", cxxopts::value<double>()->default_value("1.0"))
        ("initial-learning-rate", "Set initial learning rate", cxxopts::value<double>()->default_value("0.1"))
        ("learning-rate-shrink-factor", "Set learning rate shrink factor", cxxopts::value<double>()->default_value("0.1"))
        ("min-learning-rate", "Set minimum learning rate", cxxopts::value<double>()->default_value("1e-6"))
        ("save-interval", "Save the resulting inference network every this many steps", cxxopts::value<size_t>()->default_value("1000"))
        ("t,relative-training-length", "Relative training length", cxxopts::value<double>()->default_value("2.0"))
        ("c,cached-image-count", "Cached image count", cxxopts::value<int>()->default_value("8"))
        ("data-loader-thread-count", "Number of data loader threads", cxxopts::value<unsigned int>()->default_value(default_data_loader_thread_count.str()))
        ;

    try {
        options.parse_positional("input-directory");
        options.parse(argc, argv);

        cxxopts::check_required(options, { "input-directory" });

        std::cout << "Input directory = " << options["input-directory"].as<std::string>() << std::endl;
        std::cout << "Initial downscaling factor = " << options["initial-downscaling-factor"].as<double>() << std::endl;
        std::cout << "Further downscaling factor = " << options["further-downscaling-factor"].as<double>() << std::endl;

        if (options["initial-downscaling-factor"].as<double>() <= 0.0 || options["further-downscaling-factor"].as<double>() <= 0.0) {
            throw std::runtime_error("The downscaling factors have to be strictly positive.");
        }
    }
    catch (std::exception& e) {
        cerr << e.what() << std::endl;
        cerr << std::endl;
        cerr << options.help() << std::endl;
        return 2;
    }

    const double initial_downscaling_factor = options["initial-downscaling-factor"].as<double>();
    const double further_downscaling_factor = options["further-downscaling-factor"].as<double>();
    const bool allow_flip_upside_down = options.count("allow-flip-upside-down") > 0;
    const auto minibatch_size = options["minibatch-size"].as<size_t>();
    const auto input_dimension_multiplier = options["input-dimension-multiplier"].as<double>();
    const auto net_width_scaler = options["net-width-scaler"].as<double>();
    const auto initial_learning_rate = options["initial-learning-rate"].as<double>();
    const auto learning_rate_shrink_factor = options["learning-rate-shrink-factor"].as<double>();
    const auto min_learning_rate = options["min-learning-rate"].as<double>();
    const auto save_interval = options["save-interval"].as<size_t>();
    const auto relative_training_length = std::max(0.01, options["relative-training-length"].as<double>());
    const auto cached_image_count = options["cached-image-count"].as<int>();
    const auto data_loader_thread_count = std::max(1U, options["data-loader-thread-count"].as<unsigned int>());

    std::cout << "Allow flipping input images upside down = " << (allow_flip_upside_down ? "yes" : "no") << std::endl;
    std::cout << "Minibatch size = " << minibatch_size << std::endl;
    std::cout << "Initial learning rate = " << initial_learning_rate << std::endl;
    std::cout << "Learning rate shrink factor = " << learning_rate_shrink_factor << std::endl;
    std::cout << "Min learning rate = " << min_learning_rate << std::endl;
    std::cout << "Save interval = " << save_interval << std::endl;
    std::cout << "Relative training length = " << relative_training_length << std::endl;
    std::cout << "Cached image count = " << cached_image_count << std::endl;
    std::cout << "Data loader thread count = " << data_loader_thread_count << std::endl;

    const int required_input_dimension = NetPimpl::TrainingNet::GetRequiredInputDimension();
    std::cout << "Required input dimension = " << required_input_dimension << std::endl;

    const int requested_input_dimension = static_cast<int>(std::round(input_dimension_multiplier * required_input_dimension));
    std::cout << "Requested input dimension = " << requested_input_dimension << std::endl;

    const int actual_input_dimension = NetPimpl::RuntimeNet::GetRecommendedInputDimension(requested_input_dimension);
    std::cout << "Actual input dimension = " << actual_input_dimension << std::endl;

    const unsigned long iterations_without_progress_threshold = static_cast<unsigned long>(std::round(relative_training_length * 2000));
    const unsigned long previous_loss_values_dump_amount = static_cast<unsigned long>(std::round(relative_training_length * 400));
    const unsigned long batch_normalization_running_stats_window_size = static_cast<unsigned long>(std::round(relative_training_length * 100));

    NetPimpl::TrainingNet training_net;

    std::vector<NetPimpl::input_type> samples;
    std::vector<NetPimpl::training_label_type> labels;

    training_net.Initialize();
    training_net.SetNetWidth(net_width_scaler, 1);
    training_net.SetSynchronizationFile("annonet_trainer_state_file.dat", std::chrono::seconds(10 * 60));
    training_net.BeVerbose();
    training_net.SetLearningRate(initial_learning_rate);
    training_net.SetLearningRateShrinkFactor(learning_rate_shrink_factor);
    training_net.SetIterationsWithoutProgressThreshold(iterations_without_progress_threshold);
    training_net.SetPreviousLossValuesDumpAmount(previous_loss_values_dump_amount);
    training_net.SetAllBatchNormalizationRunningStatsWindowSizes(batch_normalization_running_stats_window_size);

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto image_files = find_image_files(options["input-directory"].as<std::string>(), true);
    cout << "images in dataset: " << image_files.size() << endl;
    if (image_files.size() == 0)
    {
        cout << "Didn't find an anno dataset. " << endl;
        return 1;
    }

    shared_lru_cache_using_std<image_filenames, std::shared_ptr<sample>, std::unordered_map> full_images_cache(
        [&](const image_filenames& image_filenames) {
            std::shared_ptr<sample> sample(new sample);
            *sample = read_sample(image_filenames, true, initial_downscaling_factor);
            return sample;
        }, cached_image_count);

    cout << endl << "Now training..." << endl;
   
    set_low_priority();

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<crop> data(2 * minibatch_size);
    auto pull_crops = [&data, &full_images_cache, &image_files, actual_input_dimension, &options](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        NetPimpl::input_type input_image;
        matrix<uint16_t> index_label_image;
        crop crop;
        randomly_crop_image_temp temp;
        while (data.is_enabled())
        {
            crop.error.clear();
            crop.warning.clear();

            const size_t index = rnd.get_random_32bit_number() % image_files.size();
            const image_filenames& image_filenames = image_files[index];
            const std::shared_ptr<sample> ground_truth_sample = full_images_cache(image_filenames);

            if (!ground_truth_sample->error.empty()) {
                crop.error = ground_truth_sample->error;
            }
            else {
                randomly_crop_image(actual_input_dimension, *ground_truth_sample, crop, rnd, options, temp);
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
        serialize("annonet.dnn") << (initial_downscaling_factor * further_downscaling_factor) << serialized.str();
    };

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
                if (warnings_already_printed.find(crop.warning) == warnings_already_printed.end()) {
                    std::cout << crop.warning << std::endl;
                    warnings_already_printed.insert(crop.warning);
                }
            }
            else {
                samples.push_back(std::move(crop.input_image_stack));
                labels.push_back(std::move(crop.target_image));
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