#pragma once

#include <vector>

int _binomial_coefficient(int n, int k);

int _sum(std::vector<int> v);

std::vector<std::vector<std::vector<int>>> _partitions(std::vector<int> v);

std::vector<std::vector<std::vector<int>>> _two_part_partitions(std::vector<int> v);

std::vector<std::vector<int>> _permutations(int size);

std::vector<std::vector<int>> _combinations(std::vector<int>, int k);

template <typename T>
std::vector<std::vector<T>> _product(std::vector<std::vector<T>> inputs) {
    auto outputs = std::vector<std::vector<T>>({{}});
    for (auto input : inputs) {
        auto new_outputs = std::vector<std::vector<T>>({});
        for (auto output : outputs) {
            for (auto i : input) {
                auto new_output = output;
                new_output.push_back(i);
                new_outputs.push_back(new_output);
            }
        }
        outputs = new_outputs;
    }
    return outputs;
}

std::vector<std::vector<int>> _product_repeat(std::vector<int> input, int repeat);

// TODO: rename
std::vector<std::vector<int>> _generate_indices(int dim, int max);
