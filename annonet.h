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

#ifndef ANNONET_H
#define ANNONET_H

#include <dlib/dnn.h>
#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include <unordered_map>

// ----------------------------------------------------------------------------------------

struct image_filenames
{
    std::string input0_filename;
    std::string input1_filename;
    std::string ground_truth_filename;
};

typedef uint8_t input_pixel_type;

struct sample
{
    int original_width = 0;
    int original_height = 0;
    image_filenames image_filenames;
    NetPimpl::input_type input_image_stack;
    NetPimpl::training_label_type target_image;
    std::string error;
};

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

sample read_sample(const image_filenames& image_filenames, bool require_ground_truth, double downscaling_factor);

void set_low_priority();

#endif // ANNONET_H