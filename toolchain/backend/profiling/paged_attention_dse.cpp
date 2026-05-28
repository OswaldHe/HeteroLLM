#include "realsys_profile.h"
#include "../../frontend/deploy/paged_attention.h"

// Helper function to create random projection weight matrix
static std::vector<std::vector<float>> create_random_projection_weight(
    size_t head_dim, size_t proj_dim, int seed = 42) {
    std::vector<std::vector<float>> weight(head_dim, std::vector<float>(proj_dim));
    srand(seed);
    for (size_t i = 0; i < head_dim; ++i) {
        for (size_t j = 0; j < proj_dim; ++j) {
            weight[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        }
    }
    return weight;
}

int main(int argc, char** argv) {
    // Default design space path
    std::string design_space_path = "design_space.json";
    
    // Parse command line arguments
    if (argc > 1) {
        design_space_path = argv[1];
    }
    
    std::cout << "Design Space Explorer for Heterogeneous LLM Serving" << std::endl;
    std::cout << "Using design space: " << design_space_path << std::endl;
    
    // Configuration parameters (matching existing test files)
    const size_t head_dim = 768;
    const size_t proj_dim = 64;
    const size_t page_size = 8;
    const size_t top_k = 64;
    
    // Create projection weight matrix
    auto projection_weight = create_random_projection_weight(head_dim, proj_dim, 42);
    
    // Instantiate the paged attention manager
    PagedAttention manager(projection_weight, page_size, top_k);
    
    // Define data creation functions for different sequence lengths
    auto create_kv_cache = [head_dim](size_t seq_len) -> data_type::KVCacheData<float> {
        data_type::KVCache<float> kv_data;
        kv_data.keys = std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim, 0.1f));
        kv_data.values = std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim, 0.1f));
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                kv_data.keys[i][j] = static_cast<float>((i * head_dim + j) % 1000) / 1000.0f;
                kv_data.values[i][j] = static_cast<float>((i * head_dim + j + 500) % 1000) / 1000.0f;
            }
        }
        return data_type::KVCacheData<float>(kv_data);
    };
    
    // Query needs to match the projected dimension (proj_dim), not head_dim
    auto create_query = [proj_dim](size_t /*seq_len*/) -> data_type::VectorQuery<float> {
        std::vector<float> query_vec(proj_dim, 1.0f);
        return data_type::VectorQuery<float>(query_vec);
    };
    
    auto create_input = [head_dim](size_t /*seq_len*/) -> data_type::VectorInputOutputData<float> {
        std::vector<float> input_vec(head_dim, 1.0f);
        return data_type::VectorInputOutputData<float>(input_vec);
    };
    
    // Run design space exploration with explicit template parameters
    auto [best_result, all_results] = explore_design_space<
        PagedAttention,
        data_type::KVCacheData<float>,
        data_type::VectorQuery<float>,
        data_type::VectorInputOutputData<float>
    >(
        manager,
        create_kv_cache,
        create_query,
        create_input,
        design_space_path,
        1,      // warmup iterations
        3,      // timing runs
        true    // verbose
    );
    
    // Print summary
    print_exploration_summary(best_result, all_results);
    
    return 0;
}
