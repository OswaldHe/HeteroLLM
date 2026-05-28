/**
 * @file realsys_profile.h
 * @brief Design space exploration testing framework (templated)
 * 
 * This file provides a framework to explore the design space defined in 
 * design_space.json. It tests all combinations of device assignments
 * (CPU/GPU/FPGA) for each pipeline step across different sequence lengths,
 * measures latency, and identifies the optimal configuration.
 * 
 * The explore_design_space function is templated to work with any
 * MemoryManager-derived class.
 */

#ifndef HETEROMM_REALSYS_PROFILE_H_
#define HETEROMM_REALSYS_PROFILE_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <functional>

// Include frontend headers
#include "../../frontend/deploy/memory_manager.h"

using namespace heteromm;
using namespace heteromm::deploy;

// ============================================================================
// JSON Parsing Utilities (simple parser for design_space.json)
// ============================================================================

/**
 * @brief Simple structure to hold the design space configuration
 */
struct DesignSpace {
    std::vector<std::string> build_memory_devices;
    std::vector<std::string> compute_score_devices;
    std::vector<std::string> memory_retrieval_devices;
    std::vector<std::string> apply_memory_devices;
    std::vector<size_t> seq_lengths;
};

/**
 * @brief Convert device string to KernelType
 */
step::KernelType string_to_kernel_type(const std::string& device) {
    if (device == "cpu") {
        return step::KernelType::CPU;
    } else if (device == "gpu") {
        return step::KernelType::GPU;
    } else if (device == "fpga") {
        return step::KernelType::FPGA;
    }
    return step::KernelType::CPU;  // Default
}

/**
 * @brief Parse a JSON array of strings (simple parser)
 */
std::vector<std::string> parse_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    
    // Find the key
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    // Find the array start
    size_t arr_start = json.find("[", key_pos);
    if (arr_start == std::string::npos) return result;
    
    // Find the array end
    size_t arr_end = json.find("]", arr_start);
    if (arr_end == std::string::npos) return result;
    
    // Extract array content
    std::string arr_content = json.substr(arr_start + 1, arr_end - arr_start - 1);
    
    // Parse individual strings
    size_t pos = 0;
    while ((pos = arr_content.find("\"", pos)) != std::string::npos) {
        size_t end_quote = arr_content.find("\"", pos + 1);
        if (end_quote != std::string::npos) {
            result.push_back(arr_content.substr(pos + 1, end_quote - pos - 1));
            pos = end_quote + 1;
        } else {
            break;
        }
    }
    
    return result;
}

/**
 * @brief Parse a JSON array of integers
 */
std::vector<size_t> parse_int_array(const std::string& json, const std::string& key) {
    std::vector<size_t> result;
    
    // Find the key
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    // Find the array start
    size_t arr_start = json.find("[", key_pos);
    if (arr_start == std::string::npos) return result;
    
    // Find the array end
    size_t arr_end = json.find("]", arr_start);
    if (arr_end == std::string::npos) return result;
    
    // Extract array content
    std::string arr_content = json.substr(arr_start + 1, arr_end - arr_start - 1);
    
    // Parse individual numbers
    std::stringstream ss(arr_content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Remove whitespace
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            result.push_back(std::stoull(token));
        }
    }
    
    return result;
}

/**
 * @brief Load design space configuration from JSON file
 */
DesignSpace load_design_space(const std::string& filepath) {
    DesignSpace ds;
    
    // Read the file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open design space file: " << filepath << std::endl;
        return ds;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();
    
    // Parse each field
    ds.build_memory_devices = parse_string_array(json_content, "build_memory");
    ds.compute_score_devices = parse_string_array(json_content, "compute_score");
    ds.memory_retrieval_devices = parse_string_array(json_content, "memory_retrieval");
    ds.apply_memory_devices = parse_string_array(json_content, "apply_memory");
    ds.seq_lengths = parse_int_array(json_content, "seq_len");
    
    return ds;
}

// ============================================================================
// Configuration and Result Structures
// ============================================================================

/**
 * @brief A single configuration point in the design space
 */
struct Configuration {
    std::string build_memory_device;
    std::string compute_score_device;
    std::string memory_retrieval_device;
    std::string apply_memory_device;
    size_t seq_len;
    
    std::string to_string() const {
        std::stringstream ss;
        ss << "seq_len=" << seq_len 
           << ", build=" << build_memory_device
           << ", score=" << compute_score_device
           << ", retrieval=" << memory_retrieval_device
           << ", apply=" << apply_memory_device;
        return ss.str();
    }
};

/**
 * @brief Result of profiling a single configuration
 */
struct ProfilingResult {
    Configuration config;
    double total_latency_ms = 0.0;
    double build_memory_latency_ms = 0.0;
    double compute_score_latency_ms = 0.0;
    double memory_retrieval_latency_ms = 0.0;
    double apply_memory_latency_ms = 0.0;
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Design Space Exploration Function (Templated)
// ============================================================================

/**
 * @brief Explore the design space and find the optimal configuration
 * 
 * This templated version works with any MemoryManager-derived class.
 * 
 * @tparam ManagerT A class derived from MemoryManager template
 * @tparam RetDataT The RetrievedData type for this manager
 * @tparam QueryT The Query type for this manager
 * @tparam InputT The Input/Output type for this manager
 * 
 * @param manager The memory manager instance to test
 * @param create_kv_cache Function to create input data for a given sequence length
 * @param create_query Function to create query for a given sequence length
 * @param create_input Function to create input for a given sequence length
 * @param design_space_path Path to the design_space.json file
 * @param num_warmup Number of warmup iterations before timing
 * @param num_runs Number of timed runs for averaging
 * @param verbose Whether to print progress
 * @return Pair of (best result, all results)
 */
template<typename ManagerT, typename RetDataT, typename QueryT, typename InputT>
std::pair<ProfilingResult, std::vector<ProfilingResult>> explore_design_space(
    ManagerT& manager,
    std::function<RetDataT(size_t seq_len)> create_kv_cache,
    std::function<QueryT(size_t seq_len)> create_query,
    std::function<InputT(size_t seq_len)> create_input,
    const std::string& design_space_path,
    int num_warmup = 1,
    int num_runs = 3,
    bool verbose = true
) {
    // Load design space
    DesignSpace ds = load_design_space(design_space_path);
    
    if (ds.seq_lengths.empty()) {
        std::cerr << "Error: No sequence lengths defined in design space" << std::endl;
        return {{}, {}};
    }
    
    std::vector<ProfilingResult> all_results;
    ProfilingResult best_result;
    best_result.total_latency_ms = std::numeric_limits<double>::max();
    
    size_t total_configs = ds.seq_lengths.size() * 
                           ds.build_memory_devices.size() *
                           ds.compute_score_devices.size() *
                           ds.memory_retrieval_devices.size() *
                           ds.apply_memory_devices.size();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Design Space Exploration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total configurations to test: " << total_configs << std::endl;
    std::cout << "Sequence lengths: ";
    for (auto sl : ds.seq_lengths) std::cout << sl << " ";
    std::cout << std::endl;
    std::cout << "Warmup iterations: " << num_warmup << std::endl;
    std::cout << "Timing runs: " << num_runs << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    size_t config_idx = 0;
    
    // Iterate through all sequence lengths
    for (size_t seq_len : ds.seq_lengths) {
        if (verbose) {
            std::cout << "\n--- Testing seq_len = " << seq_len << " ---" << std::endl;
        }
        
        // Create input data for this sequence length
        auto kv_cache = create_kv_cache(seq_len);
        auto query = create_query(seq_len);
        auto input = create_input(seq_len);
        
        // Iterate through all device assignments
        for (const auto& build_dev : ds.build_memory_devices) {
            for (const auto& compute_dev : ds.compute_score_devices) {
                for (const auto& retrieval_dev : ds.memory_retrieval_devices) {
                    for (const auto& apply_dev : ds.apply_memory_devices) {
                        config_idx++;
                        
                        Configuration config;
                        config.seq_len = seq_len;
                        config.build_memory_device = build_dev;
                        config.compute_score_device = compute_dev;
                        config.memory_retrieval_device = retrieval_dev;
                        config.apply_memory_device = apply_dev;
                        
                        // Configure the manager with this device assignment
                        PipelineKernelConfig kernel_config(
                            string_to_kernel_type(build_dev),
                            string_to_kernel_type(compute_dev),
                            string_to_kernel_type(retrieval_dev),
                            string_to_kernel_type(apply_dev)
                        );
                        manager.set_kernel_config(kernel_config);
                        
                        ProfilingResult result;
                        result.config = config;
                        result.success = true;
                        
                        // Warmup runs
                        for (int w = 0; w < num_warmup; ++w) {
                            InputT output;
                            manager.build_and_apply_memory(kv_cache, query, input, output, false);
                        }
                        
                        // Timing runs
                        double total_time = 0.0;
                        double total_build = 0.0;
                        double total_compute = 0.0;
                        double total_retrieval = 0.0;
                        double total_apply = 0.0;
                        
                        for (int r = 0; r < num_runs; ++r) {
                            InputT output;
                            auto exec_result = manager.build_and_apply_memory(
                                kv_cache, query, input, output, false);
                            
                            if (!exec_result.success) {
                                result.success = false;
                                result.error_message = exec_result.error_message;
                                break;
                            }
                            
                            total_time += exec_result.total_time_ms;
                            total_build += exec_result.build_memory_time_ms;
                            total_compute += exec_result.compute_score_time_ms;
                            total_retrieval += exec_result.retrieval_time_ms;
                            total_apply += exec_result.apply_memory_time_ms;
                        }
                        
                        if (result.success) {
                            result.total_latency_ms = total_time / num_runs;
                            result.build_memory_latency_ms = total_build / num_runs;
                            result.compute_score_latency_ms = total_compute / num_runs;
                            result.memory_retrieval_latency_ms = total_retrieval / num_runs;
                            result.apply_memory_latency_ms = total_apply / num_runs;
                        }
                        
                        all_results.push_back(result);
                        
                        // Print progress
                        if (verbose) {
                            std::cout << "[" << config_idx << "/" << total_configs << "] "
                                      << config.to_string();
                            if (result.success) {
                                std::cout << " => " << std::fixed << std::setprecision(3) 
                                          << result.total_latency_ms << " ms" << std::endl;
                            } else {
                                std::cout << " => FAILED: " << result.error_message << std::endl;
                            }
                        }
                        
                        // Update best result
                        if (result.success && result.total_latency_ms < best_result.total_latency_ms) {
                            best_result = result;
                        }
                    }
                }
            }
        }
    }
    
    return {best_result, all_results};
}

/**
 * @brief Print the summary of exploration results
 */
void print_exploration_summary(
    const ProfilingResult& best_result,
    const std::vector<ProfilingResult>& all_results
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Exploration Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Count successes and failures
    size_t success_count = 0;
    size_t failure_count = 0;
    for (const auto& r : all_results) {
        if (r.success) success_count++;
        else failure_count++;
    }
    
    std::cout << "Total configurations: " << all_results.size() << std::endl;
    std::cout << "Successful: " << success_count << std::endl;
    std::cout << "Failed: " << failure_count << std::endl;
    
    if (best_result.success) {
        std::cout << "\n--- Optimal Configuration ---" << std::endl;
        std::cout << "Sequence Length: " << best_result.config.seq_len << std::endl;
        std::cout << "Build Memory Device: " << best_result.config.build_memory_device << std::endl;
        std::cout << "Compute Score Device: " << best_result.config.compute_score_device << std::endl;
        std::cout << "Memory Retrieval Device: " << best_result.config.memory_retrieval_device << std::endl;
        std::cout << "Apply Memory Device: " << best_result.config.apply_memory_device << std::endl;
        std::cout << std::endl;
        std::cout << "--- Latency Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Build Memory: " << best_result.build_memory_latency_ms << " ms" << std::endl;
        std::cout << "Compute Score: " << best_result.compute_score_latency_ms << " ms" << std::endl;
        std::cout << "Memory Retrieval: " << best_result.memory_retrieval_latency_ms << " ms" << std::endl;
        std::cout << "Apply Memory: " << best_result.apply_memory_latency_ms << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "TOTAL LATENCY: " << best_result.total_latency_ms << " ms" << std::endl;
    } else {
        std::cout << "\nNo successful configuration found!" << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
}

#endif // HETEROMM_REALSYS_PROFILE_H_
