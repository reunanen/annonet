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
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>

#include <iostream>
#include <iterator>
#include <unordered_map>
#include <thread>

using namespace std;
using namespace dlib;

struct training_sample
{
    matrix<rgb_pixel> input_image;
    matrix<uint16_t> label_image;
    std::unordered_map<uint16_t, std::deque<point>> labeled_points_by_class;
};

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
    const training_sample& full_sample,
    training_sample& crop,
    dlib::rand& rnd
)
{
    const int dim = 115;

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
    //if (rnd.get_random_double() > 0.5) {
    //    crop.first = flipud(crop.first);
    //    crop.second = flipud(crop.second);
    //}

    // And then randomly adjust the colors.
    apply_random_color_offset(crop.input_image, rnd);
}

// ----------------------------------------------------------------------------------------

struct image_info
{
    string image_filename;
    string label_filename;
};

std::vector<image_info> get_anno_data_listing(
    const std::string& anno_data_folder
)
{
    const std::vector<file> files = get_files_in_directory_tree(anno_data_folder,
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

    std::vector<image_info> results;

    for (const file& name : files) {
        image_info image_info;
        image_info.image_filename = name;
        image_info.label_filename = name.full_name() + "_mask.png";
        std::ifstream label_file(image_info.label_filename, std::ios::binary);
        if (label_file) {
            results.push_back(image_info);
            std::cout << "Added file " << image_info.image_filename << std::endl;
        }
        else {
            std::cout << "Warning: unable to open " << image_info.label_filename << std::endl;
        }
    }

    return results;
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

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes)
{
    if (rgba_label == rgba_ignore_label) {
        return dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    }
    for (const AnnoClass& anno_class : anno_classes) {
        if (anno_class.rgba_label == rgba_label) {
            return anno_class.index;
        }
    }
    std::ostringstream error;
    error << "Unknown class: r = " << rgba_label.red << ", g = " << rgba_label.green << ", b = " << rgba_label.blue << ", alpha = " << rgba_label.alpha;
    throw std::runtime_error(error.str());
}

void decode_rgba_label_image(const dlib::matrix<dlib::rgb_alpha_pixel>& rgba_label_image, training_sample& training_sample, const std::vector<AnnoClass>& anno_classes)
{
    const long nr = rgba_label_image.nr();
    const long nc = rgba_label_image.nc();

    training_sample.label_image.set_size(nr, nc);
    training_sample.labeled_points_by_class.clear();

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            const uint16_t label = rgba_label_to_index_label(rgba_label_image(r, c), anno_classes);
            if (label != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                training_sample.labeled_points_by_class[label].push_back(point(c, r));
            }
            training_sample.label_image(r, c) = label;
        }
    }
}

// ----------------------------------------------------------------------------------------

#if 0
double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset)
{
    int num_right = 0;
    int num_wrong = 0;

    training_sample training_sample;
    matrix<rgb_pixel> rgb_label_image;

    for (const auto& image_info : dataset) {
        load_image(training_sample.input_image, image_info.image_filename);
        load_image(rgb_label_image, image_info.label_filename);

        matrix<uint16_t> net_output = anet(training_sample.input_image);

        decode_rgba_label_image(rgb_label_image, training_sample);

        const long nr = training_sample.label_image.nr();
        const long nc = training_sample.label_image.nc();

        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                const uint16_t truth = training_sample.label_image(r, c);
                if (truth != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                    const uint16_t prediction = net_output(r, c);
                    if (prediction == truth) {
                        ++num_right;
                    }
                    else {
                        ++num_wrong;
                    }
                }
            }
        }
    }

    return num_right / static_cast<double>(num_right + num_wrong);
}
#endif

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

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto anno_classes_json = read_anno_classes_file(argv[1]);
    const auto anno_classes = parse_anno_classes(anno_classes_json);
    const auto listing = get_anno_data_listing(argv[1]);
    cout << "images in dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find an anno dataset. " << endl;
        return 1;
    }

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    net_type net;
    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("annonet_trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.
    trainer.set_iterations_without_progress_threshold(20000);
    trainer.set_previous_loss_values_dump_amount(4000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(net, 1000);

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<uint16_t>> labels;

    std::vector<std::future<training_sample>> full_image_futures;
    full_image_futures.reserve(listing.size());

    const auto read_training_sample = [&anno_classes](const image_info& image_info)
    {
        training_sample training_sample;
        matrix<rgb_alpha_pixel> rgba_label_image;

        load_image(training_sample.input_image, image_info.image_filename);
        load_image(rgba_label_image, image_info.label_filename);
        decode_rgba_label_image(rgba_label_image, training_sample, anno_classes);
        
        return training_sample;
    };

    for (const image_info& image_info : listing) {
        full_image_futures.push_back(std::async(read_training_sample, image_info));
    }

    std::deque<training_sample> full_images;

    for (size_t i = 0, end = full_image_futures.size(); i < end; ++i) {
        std::cout << "\rReading image " << (i + 1) << " of " << end << "...";
        const training_sample training_sample = full_image_futures[i].get();
        if (!training_sample.labeled_points_by_class.empty()) {
            full_images.push_back(std::move(training_sample));
        }
    }

    cout << endl << "Now training..." << endl;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<training_sample> data(100);
    auto f = [&data, &full_images](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<uint16_t> index_label_image;
        training_sample temp;
        while(data.is_enabled())
        {
            const size_t index = rnd.get_random_32bit_number() % full_images.size();
            const training_sample& training_sample = full_images[index];
            randomly_crop_image(training_sample, temp, rnd);
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    size_t minibatch = 0;

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-6.
    while(trainer.get_learning_rate() >= 1e-6)
    {
        samples.clear();
        labels.clear();

        // make a 60-image mini-batch
        training_sample temp;
        while(samples.size() < 60)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.input_image));
            labels.push_back(std::move(temp.label_image));
        }

        trainer.train_one_step(samples, labels);

        if (minibatch++ % 1000 == 0) {
            trainer.get_net(force_flush_to_disk::no);
            anet_type anet = net;
            anet.clean();
            cout << "saving network" << endl;
            serialize("annonet.dnn") << anno_classes_json << anet;
        }
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    trainer.get_net();

    anet_type anet = net;
    anet.clean();
    cout << "saving network" << endl;
    serialize("annonet.dnn") << anno_classes_json << anet;

#if 0
    anet_type anet = net;

    cout << "Testing the network..." << endl;

    cout << "train accuracy  :  " << calculate_accuracy(anet, get_anno_data_listing(argv[1])) << endl;
#endif
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

