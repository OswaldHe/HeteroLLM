#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "json.hpp"

using json = nlohmann::json;

namespace heteromm {
    namespace deploy {

// sched rule: first matching
// Represents a single interval rule for build_memory
struct BuildMemoryRule {
    int retrieved_data_max;  // upper bound of retrieved_data interval (-1 means infinity)
    std::string config;      // "fpga" or "gpu"
};

// Represents a single interval rule for manage_memory_and_apply
struct ManageMemoryAndApplyRule {
    int retrieved_data_max;  // upper bound of retrieved_data interval (-1 means infinity)
    int memory_max;          // upper bound of memory interval (-1 means infinity)
    int query;
    int output;
    std::vector<std::string> config;  // e.g., ["fpga", "fpga", "gpu"]
};

// Represents a single interval rule for build_and_apply_memory
struct BuildAndApplyMemoryRule {
    int retrieved_data_max;  // upper bound of retrieved_data interval (-1 means infinity)
    int query;
    int output;
    std::vector<std::string> config;  // e.g., ["gpu", "fpga", "fpga", "gpu"]
};

// Main schedule configuration class
class ScheduleConfig {
public:
    std::vector<BuildMemoryRule> build_memory_rules;
    std::vector<ManageMemoryAndApplyRule> manage_memory_and_apply_rules;
    std::vector<BuildAndApplyMemoryRule> build_and_apply_memory_rules;

    ScheduleConfig() = default;

    // Load schedule configuration from a JSON file
    static ScheduleConfig load_from_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open schedule file: " + filepath);
        }

        json j;
        file >> j;
        file.close();

        return load_from_json(j);
    }

    // Load schedule configuration from a JSON object
    static ScheduleConfig load_from_json(const json& j) {
        ScheduleConfig config;

        // Parse build_memory rules
        if (j.contains("build_memory")) {
            for (const auto& item : j["build_memory"]) {
                BuildMemoryRule rule;
                rule.retrieved_data_max = item["retrieved_data"].get<int>();
                rule.config = item["config"].get<std::string>();
                config.build_memory_rules.push_back(rule);
            }
        }

        // Parse manage_memory_and_apply rules
        if (j.contains("manage_memory_and_apply")) {
            for (const auto& item : j["manage_memory_and_apply"]) {
                ManageMemoryAndApplyRule rule;
                rule.retrieved_data_max = item["retrieved_data"].get<int>();
                rule.memory_max = item["memory"].get<int>();
                rule.query = item["query"].get<int>();
                rule.output = item["output"].get<int>();
                for (const auto& cfg : item["config"]) {
                    rule.config.push_back(cfg.get<std::string>());
                }
                config.manage_memory_and_apply_rules.push_back(rule);
            }
        }

        // Parse build_and_apply_memory rules
        if (j.contains("build_and_apply_memory")) {
            for (const auto& item : j["build_and_apply_memory"]) {
                BuildAndApplyMemoryRule rule;
                rule.retrieved_data_max = item["retrieved_data"].get<int>();
                rule.query = item["query"].get<int>();
                rule.output = item["output"].get<int>();
                for (const auto& cfg : item["config"]) {
                    rule.config.push_back(cfg.get<std::string>());
                }
                config.build_and_apply_memory_rules.push_back(rule);
            }
        }

        return config;
    }

    // Find the appropriate build_memory config for a given retrieved_data size
    std::string get_build_memory_config(int retrieved_data) const {
        for (const auto& rule : build_memory_rules) {
            if (rule.retrieved_data_max == -1 || retrieved_data <= rule.retrieved_data_max) {
                return rule.config;
            }
        }
        throw std::runtime_error("No matching build_memory rule found");
    }

    // Find the appropriate manage_memory_and_apply config for given parameters
    std::vector<std::string> get_manage_memory_and_apply_config(int retrieved_data, int memory) const {
        for (const auto& rule : manage_memory_and_apply_rules) {
            bool match_retrieved = (rule.retrieved_data_max == -1 || retrieved_data <= rule.retrieved_data_max);
            bool match_memory = (rule.memory_max == -1 || memory <= rule.memory_max);
            if (match_retrieved && match_memory) {
                return rule.config;
            }
        }
        throw std::runtime_error("No matching manage_memory_and_apply rule found");
    }

    // Find the appropriate build_and_apply_memory config for a given retrieved_data size
    std::vector<std::string> get_build_and_apply_memory_config(int retrieved_data) const {
        for (const auto& rule : build_and_apply_memory_rules) {
            if (rule.retrieved_data_max == -1 || retrieved_data <= rule.retrieved_data_max) {
                return rule.config;
            }
        }
        throw std::runtime_error("No matching build_and_apply_memory rule found");
    }
};

    } // namespace deploy

} // namespace heteromm

#endif // UTILS_H
