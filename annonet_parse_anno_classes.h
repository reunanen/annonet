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

#ifndef ANNONET_PARSE_ANNO_CLASSES_H
#define ANNONET_PARSE_ANNO_CLASSES_H

#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------



// ----------------------------------------------------------------------------------------

struct AnnoClass {
    AnnoClass(uint16_t index, const dlib::rgb_alpha_pixel& rgba_label, const std::string& classlabel)
        : index(index), rgba_label(rgba_label), classlabel(classlabel)
    {}

    const uint16_t index = 0;
    const dlib::rgb_alpha_pixel rgba_label;
    const std::string classlabel;
};

namespace {
    dlib::rgb_alpha_pixel rgba_ignore_label(0, 0, 0, 0);
}

std::vector<AnnoClass> parse_anno_classes(const std::string& json);

#endif // ANNONET_PARSE_ANNO_CLASSES_H