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

    This part of the inference code is here in a separate header so that it's
    easy to embed even in actual applications.
*/

#ifndef ANNONET_INFER_H
#define ANNONET_INFER_H

#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include "dlib-dnn-pimpl-wrapper-for-segmentation/NetPimpl.h"
#include "tiling/tiling.h"

// Can be supplied to avoid unnecessary memory re-allocations
struct annonet_infer_temp
{
    NetPimpl::input_type input_tile;
};

struct instance_segmentation_result
{
    dlib::mmod_rect mmod_rect;
    dlib::matrix<float> segmentation_mask;
};

void annonet_infer(
    NetPimpl::RuntimeNet& net,
    std::unordered_map<std::string, SegmentationNetPimpl::RuntimeNet>& segmentation_nets_by_classlabel,
    int segmentation_target_size,
    double relative_instance_size,
    const NetPimpl::input_type& input_image,
    std::vector<instance_segmentation_result>& results,
    const std::vector<double>& gains = std::vector<double>(),
    const tiling::parameters& tiling_parameters = tiling::parameters(),
    annonet_infer_temp& temp = annonet_infer_temp()
);

#endif // ANNONET_INFER_H
