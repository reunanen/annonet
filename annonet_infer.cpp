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

#include "annonet_infer.h"
#include "annonet.h"
#include <dlib/dnn.h>
#include "tuc/include/tuc/numeric.hpp"
#include <unordered_set>

size_t tensor_index(const dlib::tensor& t, long sample, long k, long row, long column)
{
    // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
    return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
}

unsigned long annonet_infer(
    NetPimpl::RuntimeNet& net,
    const NetPimpl::input_type& input_image,
    int input_dimension,
    const std::vector<double>& gains,
    annonet_infer_temp& temp
)
{
    // TODO: for variation, produce different versions and feed-forward them all
    //       final result can be a vote, or something like that
    //       note: need also confidence, or similar output

    convert_for_processing(input_image, temp.input_image, input_dimension);

    const auto& output_tensor = net.Forward(temp.input_image);

    const float* out = output_tensor.host();

    int r = 0;
    int c = 0;

    uint16_t label = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    float max_value = -std::numeric_limits<float>::infinity();
    for (long k = 0; k < output_tensor.k(); ++k)
    {
        const double gain = gains.empty() ? 0.0 : gains[k];
        const float value = out[tensor_index(output_tensor, 0, k, r, c)] + gain;
        if (value > max_value)
        {
            label = static_cast<uint16_t>(k);
            max_value = value;
        }
    }
    return label;
}
