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

    This part of the inference code is here in a separate header so that it's
    easy to embed even in actual applications.
*/

#ifndef ANNONET_INFER_H
#define ANNONET_INFER_H

#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include "tiling/tiling.h"

// Can be supplied to avoid unnecessary memory re-allocations
struct annonet_infer_temp
{
    NetPimpl::input_type input_tile;
    std::vector<dlib::point> detection_seeds;
    dlib::matrix<unsigned int> connected_blobs;
};

// returns p
double annonet_infer(
    NetPimpl::RuntimeNet& net,
    const NetPimpl::input_type& input_image,
    dlib::matrix<uint16_t>& result_image,
    const std::vector<double>& gains = std::vector<double>(),
    const std::vector<double>& detection_levels = std::vector<double>(),
    const tiling::parameters& tiling_parameters = tiling::parameters(),
    annonet_infer_temp& temp = annonet_infer_temp()
);

#endif // ANNONET_INFER_H
