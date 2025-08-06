// ---------- NgramPredictorTrie.hpp -----------------------------------------
#pragma once
#include <queue>
#include <vector>
#include <string>
#include <utility>
#include <trie_prob_lm.hpp>

namespace tongrams {

/*  Thin wrapper that exposes predict().
    Model     –  a concrete tongrams::trie_prob_lm<...>
*/
template <typename Model>
class TriePredictor {
public:
    explicit TriePredictor(const std::string& index_path)
        : model_(index_path)                                    // load index.bin
    {
        // --------  cache full vocabulary for fast mapping  ----------
        uint64_t V = model_.vocab_size();                       // helper in util.hpp
        vocab_bytes_.reserve(V);
        vocab_strings_.reserve(V);
        for (uint64_t id = 0; id < V; ++id) {
            auto br = model_.vocab_byte_range(id);              // start, end ptr
            vocab_bytes_.push_back(br);
            vocab_strings_.emplace_back(br.first, br.second);
        }
        for (uint64_t id = 0; id < vocab_strings_.size(); ++id)
            vocab_map_.emplace(vocab_strings_[id], id);
    }

    /*  Return top-k predictions given a *prepared* state (context)        */
    std::vector<std::pair<std::string,float>>
    predict(const typename Model::state_type& ctx_state,
            std::size_t k = 5) const
    {
        using Item = std::pair<float,uint64_t>;                 // logP , word-id
        auto cmp = [](Item a, Item b){ return a.first > b.first; };
        std::priority_queue<Item,std::vector<Item>,decltype(cmp)> heap(cmp);

        // 1) find the child range for this context  ---------------------
        uint8_t hlen = ctx_state.length;                        // 0 … N-1
        auto r = child_range(ctx_state);                 // helper below
        if (r.begin == r.end) return {};                        // no successors

        // 2) score each successor  --------------------------------------
        for (auto it = r.begin; it != r.end; ++it) {
            uint64_t child_id = *it;

            // Build a scratch copy of state so original remains unchanged
            auto tmp = ctx_state;
            bool oov = false;
            float lp = model_.score(tmp, vocab_bytes_[child_id], oov);

            if (heap.size() < k) heap.push({lp,child_id});
            else if (lp > heap.top().first) { heap.pop(); heap.push({lp,child_id}); }
        }

        // 3) export in best-first order  --------------------------------
        std::vector<std::pair<std::string,float>> out(heap.size());
        for (std::size_t i = out.size(); i--;) {
            out[i] = { vocab_strings_[heap.top().second], heap.top().first };
            heap.pop();
        }
        return out;
    }

    /* Convenience: advance state with a word string */
    void feed(typename Model::state_type& st, const std::string& w) const {
        auto it = vocab_map_.find(w);
        if (it == vocab_map_.end()) return;        // OOV: ignore or handle differently
        bool oov=false; model_.score(st, vocab_bytes_[it->second], oov);
    }

    const Model&  model() const { return model_; }
          Model&  model()       { return model_; }

private:
    // child_range() helper: returns [begin,end) of 1-level successors
    typename Model::array_type::range_type
    child_range(const typename Model::state_type& st) const
    {
        if (st.length == 0) return {nullptr,nullptr};           // empty context
        // last added word is at words.back() (ring buffer logic)
        uint64_t id = *(st.words.rbegin());
        return model_.array(0).range(id);                       // order-1 children
    }

    Model model_;
    std::vector<byte_range>         vocab_bytes_;
    std::vector<std::string>        vocab_strings_;
    std::unordered_map<std::string,uint64_t> vocab_map_;
};

} // namespace tongrams
// --------------------------------------------------------------------------

/* ---------------- Example usage ------------------------------------------
#include "NgramPredictorTrie.hpp"
using LM = tongrams::trie_prob_lm<...>;        // your concrete template args
tongrams::TriePredictor<LM> P("index.bin");

auto st = P.model().state();
P.feed(st,"the"); P.feed(st,"quick"); P.feed(st,"brown");
auto best = P.predict(st, 3);
for (auto& [w,p] : best) std::cout << w << "  " << std::exp(p) << '\n';
-------------------------------------------------------------------------- */
