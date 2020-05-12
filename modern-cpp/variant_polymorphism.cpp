// https://www.bfilipek.com/2020/04/variant-virtual-polymorphism.html
#include <iostream>
#include <variant>
#include <vector>

struct VSimpleLabel {
    std::string _str;
};

struct VDateLabel {
    std::string _str;
};

struct VIconLabel {
    std::string _str;
    std::string _iconSrc;
};

struct HTMLLabelBuilder {
    [[nodiscard]] std::string operator()(const VSimpleLabel &label) {
      return "<p>" + label._str + "</p>";
    }

    [[nodiscard]] std::string operator()(const VDateLabel &label) {
      return "<p class=\"date\">Date: " + label._str + "</p>";
    }

    [[nodiscard]] std::string operator()(const VIconLabel &label) {
      return "<p><img src=\"" +
             label._iconSrc + "\"/>" + label._str + "</p>";
    }
};

int main() {
  using LabelVariant = std::variant<VSimpleLabel, VDateLabel, VIconLabel>;

  std::vector<LabelVariant> vecLabels;
  vecLabels.emplace_back(VSimpleLabel{"Hello World"});
  vecLabels.emplace_back(VDateLabel{"10th August 2020"});
  vecLabels.emplace_back(VIconLabel{"Error", "error.png"});

  std::string finalHTML;
  for (auto &label : vecLabels)
    finalHTML += std::visit(HTMLLabelBuilder{}, label) + '\n';

  std::cout << finalHTML;
}