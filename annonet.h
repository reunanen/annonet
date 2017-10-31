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

// ----------------------------------------------------------------------------------------

inline bool operator == (const dlib::rgb_pixel& a, const dlib::rgb_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue;
}

// ----------------------------------------------------------------------------------------

struct AnnoClass {
    AnnoClass(uint16_t index, const dlib::rgb_pixel& rgb_label, const char* classlabel)
        : index(index), rgb_label(rgb_label), classlabel(classlabel)
    {}

    const uint16_t index = 0;
    const dlib::rgb_pixel rgb_label;
    const char* classlabel = nullptr;
};

namespace {
    constexpr int class_count = 2;

    const std::vector<AnnoClass> classes = {
        AnnoClass(0, dlib::rgb_pixel(0, 63, 0), "clean"),
        AnnoClass(1, dlib::rgb_pixel(127, 0, 0), "defect"),
        AnnoClass(dlib::loss_multiclass_log_per_pixel_::label_to_ignore, dlib::rgb_pixel(0, 0, 0), "ignore"),
    };
}

template <typename Predicate>
const AnnoClass& find_anno_class(Predicate predicate)
{
    const auto i = std::find_if(classes.begin(), classes.end(), predicate);

    if (i != classes.end()) {
        return *i;
    }
    else {
        throw std::runtime_error("Unable to find a matching anno class");
    }
}

// ----------------------------------------------------------------------------------------

#ifndef __INTELLISENSE__

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block = BN<dlib::con<N,3,3,1,1, dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using blockt = BN<dlib::cont<N,3,3,1,1,dlib::relu<BN<dlib::cont<N,3,3,stride,stride,SUBNET>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N,2,2,2,2,dlib::skip1<dlib::tag2<blockt<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res       = dlib::relu<residual<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = dlib::relu<residual_down<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_up    = dlib::relu<residual_up<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_up   = dlib::relu<residual_up<block,N,dlib::affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

#if 0
template <typename SUBNET> using level1 = res_down<512, SUBNET>;
template <typename SUBNET> using level2 = res_down<256, SUBNET>;
template <typename SUBNET> using level3 = res_down<128, SUBNET>;
template <typename SUBNET> using level4 = res<64, SUBNET>;

template <typename SUBNET> using alevel1 = ares_down<512, SUBNET>;
template <typename SUBNET> using alevel2 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel3 = ares_down<128, SUBNET>;
template <typename SUBNET> using alevel4 = ares<64, SUBNET>;

template <typename SUBNET> using level1t = res_up<512, SUBNET>;
template <typename SUBNET> using level2t = res_up<256, SUBNET>;
template <typename SUBNET> using level3t = res_up<128, SUBNET>;
template <typename SUBNET> using level4t = res_up<64, SUBNET>;

template <typename SUBNET> using alevel1t = ares_up<512, SUBNET>;
template <typename SUBNET> using alevel2t = ares_up<256, SUBNET>;
template <typename SUBNET> using alevel3t = ares_up<128, SUBNET>;
template <typename SUBNET> using alevel4t = ares_up<64, SUBNET>;
#endif

#if 1
template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128,res<128,res<128,res_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

template <typename SUBNET> using level1t = res<512,res<512,res_up<512,SUBNET>>>;
template <typename SUBNET> using level2t = res<256,res<256,res<256,res<256,res<256,res_up<256,SUBNET>>>>>>;
template <typename SUBNET> using level3t = res<128,res<128,res<128,res_up<128, SUBNET>>>>;
template <typename SUBNET> using level4t = res<64,res<64,res_up<64,SUBNET>>>;

template <typename SUBNET> using alevel1t = ares<512,ares<512,ares_up<512,SUBNET>>>;
template <typename SUBNET> using alevel2t = ares<256,ares<256,ares<256,ares<256,ares<256,ares_up<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3t = ares<128,ares<128,ares<128,ares_up<128,SUBNET>>>>;
template <typename SUBNET> using alevel4t = ares<64,ares<64,ares_up<64,SUBNET>>>;
#endif

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::bn_con<dlib::cont<class_count,7,7,2,2,
                            level4t<level3t</*level2t<level1t<
                            level1<level2<*/level3<level4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>/*>>>>*/;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::affine<dlib::cont<class_count,7,7,2,2,
                            alevel4t<alevel3t</*alevel2t<alevel1t<
                            alevel1<alevel2<*/alevel3<alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>/*>>>>*/;

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_