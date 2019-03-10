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
*/

#include "annonet_parse_anno_classes.h"
#include <rapidjson/document.h>

// ----------------------------------------------------------------------------------------

bool operator == (const dlib::rgb_alpha_pixel& a, const dlib::rgb_alpha_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue && a.alpha == b.alpha;
}

// ----------------------------------------------------------------------------------------

std::vector<AnnoClass> parse_anno_classes(const std::string& json)
{
    if (json.empty()) {
        // Use the default anno classes
        return std::vector<AnnoClass>{
            AnnoClass(0, dlib::rgb_alpha_pixel(127, 127, 127, 128), "<<ignore>>"),
            AnnoClass(1, dlib::rgb_alpha_pixel(255, 255, 0, 128), "minor defect"),
            AnnoClass(2, dlib::rgb_alpha_pixel(255, 0, 0, 128), "major defect"),
        };
    }

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        throw std::runtime_error("Error parsing json\n" + json);
    }

    if (!doc.IsObject()) {
        throw std::runtime_error("Unexpected anno classes json content - the document should be an object");
    }

    const auto anno_classes_member = doc.FindMember("anno_classes");

    if (anno_classes_member == doc.MemberEnd() || !anno_classes_member->value.IsArray()) {
        throw std::runtime_error("Unexpected anno classes json content - there should be an anno_classes array");
    }

    std::vector<AnnoClass> anno_classes;

    for (rapidjson::SizeType i = 0, end = anno_classes_member->value.Size(); i < end; ++i) {
        const auto& anno_class = anno_classes_member->value[i];
        const auto name_member = anno_class.FindMember("name");
        const auto color_member = anno_class.FindMember("color");
        if (name_member == anno_class.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - no name found");
        }
        if (color_member == anno_class.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - no color found");
        }
        const auto& color = color_member->value;
        const auto red_member = color.FindMember("red");
        const auto green_member = color.FindMember("green");
        const auto blue_member = color.FindMember("blue");
        const auto alpha_member = color.FindMember("alpha");
        if (red_member == color.MemberEnd() || green_member == color.MemberEnd() || blue_member == color.MemberEnd() || alpha_member == color.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - color should have all components (red, green, blue, alpha)");
        }
        dlib::rgb_alpha_pixel rgba_value(
            red_member->value.GetInt(),
            green_member->value.GetInt(),
            blue_member->value.GetInt(),
            alpha_member->value.GetInt()
        );

        if (rgba_value == rgba_ignore_label) {
            throw std::runtime_error("Unexpected anno classes json content - rgba (0, 0, 0, 0) is reserved for pixels to be ignored");
        }

        anno_classes.push_back(AnnoClass(i, rgba_value, name_member->value.GetString()));
    }

    return anno_classes;
}
