#pragma once

#include <cstddef>
#include <vector>

namespace anc {
    // Fixed-length history of the most recent samples, newest first.
    //
    // at(0) is the sample pushed by the latest push(), at(k) the one pushed
    // k calls earlier. Before `length` pushes, the missing samples read as zero,
    // which implements the zero initial state of the specification (section 2).
    //
    // push() and at() never allocate; only resize() does.
    class History {
        public:
            explicit History(std::size_t length = 0) { resize(length); }

            void resize(std::size_t length);
            void clear();
            void push(float value);
            float at(std::size_t k) const;

            std::size_t size() const { return data_.size(); }
        private:
            std::vector<float> data_;
            std::size_t newest_ = 0;
        };      
}   // namespace anc