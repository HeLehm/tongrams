// test/test_predict_demo.cpp
#include <iostream>

#include "utils/util.hpp"
#include "lm_types.hpp"
#include "../external/essentials/include/essentials.hpp"
#include "../external/cmd_line_parser/include/parser.hpp"
#include <boost/preprocessor/seq/for_each.hpp>

#include "NgramPredictor.hpp"

using namespace tongrams;

/* demo context ----------------------------------------------------------- */
static const std::vector<std::string> kContext = {
    "the", "quick", "brown"
};

template <typename Model>
void run_demo(Model& model, std::size_t k)
{
    TriePredictor<Model> P(model);
    auto st = model.state();
    for (auto& w : kContext) P.feed(st, w);

    auto best = P.predict(st, k);

    std::cout << "Context:"; for (auto& w : kContext) std::cout << ' ' << w;
    std::cout << "\nTop-" << k << " predictions\n";
    for (auto& [w,p] : best)
        std::cout << "  " << w << "\tlogP=" << p << '\n';
}

int main(int argc, char** argv)
{
    cmd_line_parser::parser cli(argc, argv);
    cli.add("binary_filename", "Binary filename.");
    cli.add("k", "How many suggestions");
    if (!cli.parse()) return 1;

    std::string bin  = cli.get<std::string>("binary_filename");
    std::size_t k    = cli.get<int>("k");
    std::string type = util::get_model_type(bin.c_str());

    if (false) {}   // dummy branch for BOOST_PP trick

#define LOOP_BODY(R, DATA, T)                                                 \
    else if (type == BOOST_PP_STRINGIZE(T)) {                                 \
        T model;                                                              \
        essentials::logger("Loading data structure");                         \
        util::load(model, bin.c_str());                                       \
        run_demo<T>(model, k);                                                \
    }

    BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, TONGRAMS_TRIE_PROB_TYPES)

#undef LOOP_BODY
    else {
        std::cerr << "Error: model type '" << type << "' is not supported by demo.\n";
        return 1;
    }
    return 0;
}
