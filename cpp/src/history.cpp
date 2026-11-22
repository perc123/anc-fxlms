#include "anc/history.hpp"

#include <algorithm>

namespace anc
{
    void History::resize(std::size_t length) {
        data_.assign(length, 0.0f);
        newest_ = 0;
    }

    void History::clear() {
        std::fill(data_.begin(), data_.end(), 0.0f);
        newest_ = 0;
    }

    void History::push(float value) {
        newest_ = (newest_ == 0) ? data_.size() -1 : newest_ - 1;
        data_[newest_] = value;
    }

    float History::at(std::size_t k) const {
        std::size_t index = newest_ + k;
        if (index >= data_.size()) {
            index -= data_.size();
        }
        return data_[index];
    }
    
} // namespace anc