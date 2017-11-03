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
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>

#include <iostream>
#include <iterator>
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

void randomly_crop_image (
    int dim,
    const sample& full_sample,
    sample& crop,
    dlib::rand& rnd
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
    extract_image_chip(full_sample.label_image, chip_details, crop.label_image, interpolate_nearest_neighbor());

    // Also randomly flip the input image and the labels.
    if (rnd.get_random_double() > 0.5) {
        //crop.input_image = flipud(crop.input_image);
        //crop.label_image = flipud(crop.label_image);
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
    if (argc != 2)
    {
        cout << "To run this program you need data annotated using the anno program." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./annonet_train /path/to/anno/data" << endl;
        return 1;
    }

    const int required_input_dimension = NetPimpl::TrainingNet::GetRequiredInputDimension();
    std::cout << "Required input dimension = " << required_input_dimension << std::endl;

    const auto anno_classes_json = read_anno_classes_file(argv[1]);
    const auto anno_classes = parse_anno_classes(anno_classes_json);

    const double initial_learning_rate = 0.1;
    const double learning_rate_shrink_factor = 0.1;
    const double min_learning_rate = 1e-6;
    const unsigned long iterations_without_progress_threshold = 20000;

    NetPimpl::TrainingNet training_net;
    training_net.Initialize();
    training_net.SetClassCount(anno_classes.size());
    training_net.SetLearningRate(initial_learning_rate);
    training_net.SetLearningRateShrinkFactor(learning_rate_shrink_factor);
    training_net.SetIterationsWithoutProgressThreshold(iterations_without_progress_threshold);
    training_net.SetSynchronizationFile("annonet_trainer_state_file.dat", std::chrono::seconds(10 * 60));
    training_net.BeVerbose();

    // TODO
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    //set_all_bn_running_stats_window_sizes(net, 1000);

    std::vector<matrix<input_pixel_type>> samples;
    std::vector<matrix<uint16_t>> labels;

    { // Test that the input size is correct for the net that we have built
        for (uint16_t label = 0; label < anno_classes.size(); ++label) {
            matrix<input_pixel_type> input_image(required_input_dimension, required_input_dimension);
            matrix<uint16_t> label_image(required_input_dimension, required_input_dimension);
            input_image = 0;
            label_image = label;

            samples.push_back(std::move(input_image));
            labels.push_back(std::move(label_image));
        }

        training_net.StartTraining(samples, labels);
    }

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto image_files = find_image_files(argv[1], true);
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
    
    for (unsigned int i = 0, end = std::thread::hardware_concurrency(); i < end; ++i) {
        full_image_readers.push_back(std::thread([&]() {
            image_filenames image_filenames;
            while (full_image_read_requests.dequeue(image_filenames)) {
                full_image_read_results.enqueue(read_sample(image_filenames, anno_classes, true));
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

    const size_t minibatchSize = 200;
    const size_t saveInterval = 1000;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<sample> data(2 * minibatchSize);
    auto pull_crops = [&data, &full_images, required_input_dimension](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<input_pixel_type> input_image;
        matrix<uint16_t> index_label_image;
        sample temp;
        while (data.is_enabled())
        {
            const size_t index = rnd.get_random_32bit_number() % full_images.size();
            const sample& ground_truth_sample = full_images[index];
            randomly_crop_image(required_input_dimension, ground_truth_sample, temp, rnd);
            data.enqueue(temp);
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
        serialize("annonet.dnn") << anno_classes_json << serialized.str();
    };

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    while (training_net.GetLearningRate() >= min_learning_rate)
    {
        samples.clear();
        labels.clear();

        // make a mini-batch
        sample temp;
        while(samples.size() < minibatchSize)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.input_image));
            labels.push_back(std::move(temp.label_image));
        }

        training_net.StartTraining(samples, labels);

        if (minibatch++ % saveInterval == 0) {
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