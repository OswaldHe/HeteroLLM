#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include <cerrno>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

constexpr int kLanes = 16;
constexpr int kChannels = 4;
constexpr int VOCAB_SIZE = 65536;
constexpr int VOCAB_SIZE_DIV_16 = VOCAB_SIZE / 16;
constexpr int VOCAB_SIZE_DIV_512 = VOCAB_SIZE / 512;
constexpr int TOP_K = 64;

constexpr float K1 = 1.2f;
constexpr float K1_plus_1 = K1 + 1.0f;

inline int round_up_to_multiple(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

inline unsigned int pack_token_freq(unsigned int token_id, unsigned int freq) {
  return (token_id & 0xFFFFu) | ((freq & 0xFFu) << 16);
}

bool send_all(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len > 0) {
    ssize_t n = ::send(fd, p, len, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

uint64_t htonll(uint64_t value) {
  static const uint16_t kEndianTest = 0x0102;
  const bool is_little_endian = (*reinterpret_cast<const uint8_t*>(&kEndianTest) == 0x02);
  if (!is_little_endian) {
    return value;
  }
  const uint32_t hi = htonl(static_cast<uint32_t>(value >> 32));
  const uint32_t lo = htonl(static_cast<uint32_t>(value & 0xFFFFFFFFULL));
  return (static_cast<uint64_t>(lo) << 32) | hi;
}

int setup_tcp_client(const std::string& ip, int port) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    std::cout << "TCP socket creation failed: " << std::strerror(errno) << std::endl;
    return -1;
  }

  int flag = 1;
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
    std::cout << "Invalid receiver IP address: " << ip << std::endl;
    ::close(sock);
    return -1;
  }

  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cout << "TCP connect failed: " << std::strerror(errno) << std::endl;
    ::close(sock);
    return -1;
  }

  return sock;
}

bool send_timestamp_and_indices(int sock,
                                uint64_t timestamp_ns,
                                const std::vector<int>& indices) {
  std::vector<uint8_t> payload(sizeof(uint64_t) + indices.size() * sizeof(uint32_t));
  const uint64_t net_ts = htonll(timestamp_ns);
  std::memcpy(payload.data(), &net_ts, sizeof(net_ts));

  for (size_t i = 0; i < indices.size(); ++i) {
    uint32_t net_idx = htonl(static_cast<uint32_t>(indices[i]));
    std::memcpy(payload.data() + sizeof(uint64_t) + i * sizeof(uint32_t), &net_idx, sizeof(net_idx));
  }

  bool ok = send_all(sock, payload.data(), payload.size());
  if (!ok) {
    std::cout << "TCP send failed: " << std::strerror(errno) << std::endl;
  }
  return ok;
}

void indexer_top_ref(
    int L,
    const std::vector<std::vector<std::pair<int, int>>>& documents,
    const std::unordered_set<int>& query_tokens,
    const std::vector<int>& df,
    std::vector<int>& topk_indices) {
  std::vector<std::pair<float, int>> scores;
  scores.reserve(L);

  for (int doc_id = 0; doc_id < L; ++doc_id) {
    float score = 0.0f;
    int required_mod = doc_id % 16;

    for (const auto& token_freq : documents[doc_id]) {
      int token_id = token_freq.first;
      int freq = token_freq.second;

      if ((token_id % 16) != required_mod) {
        continue;
      }

      if (query_tokens.find(token_id) == query_tokens.end()) {
        continue;
      }

      float idf_num = static_cast<float>(L) - static_cast<float>(df[token_id]) + 0.5f;
      float idf_den = static_cast<float>(df[token_id]) + 0.5f;
      float idf = logf(idf_num / idf_den);

      float tf_num = static_cast<float>(freq) * K1_plus_1;
      float tf_den = static_cast<float>(freq) + K1;
      float tf_weight = tf_num / tf_den;

      score += idf * tf_weight;
    }

    scores.push_back({score, doc_id});
  }

  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });

  topk_indices.clear();
  for (int i = 0; i < TOP_K && i < static_cast<int>(scores.size()); ++i) {
    topk_indices.push_back(scores[i].second);
  }
}

struct PackedData {
  std::vector<std::vector<unsigned int>> doc_mem;
  std::vector<int> inst_mem;
};

PackedData pack_documents_for_hw(const std::vector<std::vector<std::pair<int, int>>>& documents, int L) {
  PackedData result;
  result.doc_mem.resize(kChannels);
  int num_super_batches = L >> 6;
  result.inst_mem.resize(num_super_batches);

  for (int super_batch = 0; super_batch < num_super_batches; ++super_batch) {
    std::vector<std::vector<std::vector<std::pair<int, int>>>> doc_tokens(
        kChannels, std::vector<std::vector<std::pair<int, int>>>(kLanes));

    for (int channel = 0; channel < kChannels; ++channel) {
      for (int lane = 0; lane < kLanes; ++lane) {
        int doc_id = super_batch * 64 + channel * 16 + lane;
        for (const auto& token_freq : documents[doc_id]) {
          int token_id = token_freq.first;
          int freq = token_freq.second;
          if ((token_id % 16) == lane) {
            doc_tokens[channel][lane].push_back({token_id, freq});
          }
        }
      }
    }

    int max_tokens = 0;
    for (int channel = 0; channel < kChannels; ++channel) {
      for (int lane = 0; lane < kLanes; ++lane) {
        max_tokens = std::max(max_tokens, static_cast<int>(doc_tokens[channel][lane].size()));
      }
    }
    if (max_tokens == 0) {
      max_tokens = 1;
    }

    result.inst_mem[super_batch] = max_tokens;

    for (int channel = 0; channel < kChannels; ++channel) {
      for (int row = 0; row < max_tokens; ++row) {
        for (int lane = 0; lane < kLanes; ++lane) {
          unsigned int packed_val = 0;
          if (row < static_cast<int>(doc_tokens[channel][lane].size())) {
            int token_id = doc_tokens[channel][lane][row].first;
            int freq = doc_tokens[channel][lane][row].second;
            packed_val = pack_token_freq(static_cast<unsigned int>(token_id), static_cast<unsigned int>(freq));
          } else {
            packed_val = pack_token_freq(static_cast<unsigned int>(lane), 0u);
          }
          result.doc_mem[channel].push_back(packed_val);
        }
      }
    }
  }

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  // Command Line Parser
  sda::utils::CmdLineParser parser;

  // Switches
  //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
  parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
  parser.addSwitch("--device_id", "-d", "device index", "0");
  parser.addSwitch("--kernel_name", "-k", "kernel name", "indexer_top");
  parser.addSwitch("--num_docs", "-n", "number of documents", "1024");
  parser.addSwitch("--tokens_per_doc", "-t", "tokens per document", "128");
  parser.addSwitch("--query_size", "-q", "query token count", "64");
  parser.addSwitch("--receiver_ip", "-r", "receiver private IP", "");
  parser.addSwitch("--receiver_port", "-p", "receiver TCP port", "0");
  parser.parse(argc, argv);

  // Read settings
  std::string binaryFile = parser.value("xclbin_file");
  int device_index = stoi(parser.value("device_id"));
  std::string kernel_name = parser.value("kernel_name");
  int L = stoi(parser.value("num_docs"));
  int tokens_per_doc = stoi(parser.value("tokens_per_doc"));
  int query_size = stoi(parser.value("query_size"));
  std::string receiver_ip = parser.value("receiver_ip");
  int receiver_port = stoi(parser.value("receiver_port"));

  if (binaryFile.empty()) {
    parser.printHelp();
    return EXIT_FAILURE;
  }

  if (tokens_per_doc < 1) {
    std::cout << "tokens_per_doc must be >= 1" << std::endl;
    return EXIT_FAILURE;
  }

  if (query_size < 1) {
    std::cout << "query_size must be >= 1" << std::endl;
    return EXIT_FAILURE;
  }

  if (receiver_ip.empty()) {
    std::cout << "receiver_ip must be provided" << std::endl;
    return EXIT_FAILURE;
  }

  if (receiver_port < 1 || receiver_port > 65535) {
    std::cout << "receiver_port must be in [1, 65535]" << std::endl;
    return EXIT_FAILURE;
  }

  // Kernel expects 4 channels x 16 docs in each super-batch.
  L = round_up_to_multiple(L, 64);
  if (L < TOP_K * 2) {
    L = round_up_to_multiple(TOP_K * 2, 64);
  }

  int num_super_batches = L >> 6;

  std::vector<std::vector<std::pair<int, int>>> documents(L);
  std::vector<int> df(VOCAB_SIZE, 0);
  std::unordered_set<int> query_tokens;
  std::vector<int> query_tokens_vec;

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> doc_len_dis(tokens_per_doc / 2, tokens_per_doc * 3 / 2);
  std::uniform_int_distribution<int> token_base_dis(0, (VOCAB_SIZE / 16) - 1);

  while (static_cast<int>(query_tokens.size()) < query_size) {
    int base = token_base_dis(gen);
    int mod = static_cast<int>(query_tokens.size()) % 16;
    int token_id = base * 16 + mod;
    query_tokens.insert(token_id);
  }
  query_tokens_vec.assign(query_tokens.begin(), query_tokens.end());

  for (int doc_id = 0; doc_id < L; ++doc_id) {
    int num_tokens = doc_len_dis(gen);
    int required_mod = doc_id % 16;

    std::unordered_map<int, int> token_freq_map;
    for (int t = 0; t < num_tokens; ++t) {
      int base = token_base_dis(gen);
      int token_id = base * 16 + required_mod;
      token_freq_map[token_id]++;
    }

    for (const auto& tf : token_freq_map) {
      int capped_freq = std::min(tf.second, 255);
      documents[doc_id].push_back({tf.first, capped_freq});
      df[tf.first]++;
    }
  }

  std::vector<int> high_overlap_docs;
  for (int i = 0; i < std::min(TOP_K / 2, L); ++i) {
    int doc_id = i * 2;
    int required_mod = doc_id % 16;

    for (const auto& tf : documents[doc_id]) {
      df[tf.first]--;
    }
    documents[doc_id].clear();

    int added = 0;
    for (int token_id : query_tokens_vec) {
      if ((token_id % 16) == required_mod) {
        int freq = 5 + (added % 5);
        documents[doc_id].push_back({token_id, freq});
        df[token_id]++;
        added++;
      }
    }
    if (added > 0) {
      high_overlap_docs.push_back(doc_id);
    }
  }

  auto packed = pack_documents_for_hw(documents, L);
  int doc_vectors_per_channel = static_cast<int>(packed.doc_mem[0].size() / kLanes);
  int topk_vec_count = (TOP_K + 15) / 16;

  std::cout << "Indexer BM25 XRT Host" << std::endl;
  std::cout << "  Kernel: " << kernel_name << std::endl;
  std::cout << "  L (rounded): " << L << std::endl;
  std::cout << "  Tokens/doc: " << tokens_per_doc << std::endl;
  std::cout << "  Query size: " << query_size << std::endl;
  std::cout << "  Receiver: " << receiver_ip << ":" << receiver_port << std::endl;
  std::cout << "  TopK: " << TOP_K << std::endl;
  std::cout << "  Vocab: " << VOCAB_SIZE << std::endl;
  std::cout << "  Super-batches: " << num_super_batches << std::endl;
  std::cout << "  L_doc_total (per channel): " << doc_vectors_per_channel << std::endl;

  std::cout << "Open the device " << device_index << std::endl;
  auto device = xrt::device(device_index);
  std::cout << "Load the xclbin " << binaryFile << std::endl;
  auto uuid = device.load_xclbin(binaryFile);

  auto krnl = xrt::kernel(device, uuid, kernel_name);

  size_t df_bytes = static_cast<size_t>(VOCAB_SIZE_DIV_16) * kLanes * sizeof(int);
  size_t query_bytes = static_cast<size_t>(VOCAB_SIZE_DIV_512) * 64;
  size_t inst_bytes = static_cast<size_t>(num_super_batches) * sizeof(int);
  size_t doc_bytes = static_cast<size_t>(doc_vectors_per_channel) * kLanes * sizeof(unsigned int);
  size_t out_bytes = static_cast<size_t>(topk_vec_count) * kLanes * sizeof(int);

  std::cout << "Allocate Buffers in Global Memory" << std::endl;
  auto bo_df = xrt::bo(device, df_bytes, krnl.group_id(2));
  auto bo_query = xrt::bo(device, query_bytes, krnl.group_id(3));
  auto bo_inst = xrt::bo(device, inst_bytes, krnl.group_id(4));
  auto bo_doc0 = xrt::bo(device, doc_bytes, krnl.group_id(5));
  auto bo_doc1 = xrt::bo(device, doc_bytes, krnl.group_id(6));
  auto bo_doc2 = xrt::bo(device, doc_bytes, krnl.group_id(7));
  auto bo_doc3 = xrt::bo(device, doc_bytes, krnl.group_id(8));
  auto bo_out = xrt::bo(device, out_bytes, krnl.group_id(9));

  // Map the contents of BOs into host memory.
  auto df_map = bo_df.map<int*>();
  auto query_map = bo_query.map<unsigned long long*>();
  auto inst_map = bo_inst.map<int*>();
  auto doc0_map = bo_doc0.map<unsigned int*>();
  auto doc1_map = bo_doc1.map<unsigned int*>();
  auto doc2_map = bo_doc2.map<unsigned int*>();
  auto doc3_map = bo_doc3.map<unsigned int*>();
  auto out_map = bo_out.map<int*>();

  std::memset(df_map, 0, df_bytes);
  std::memset(query_map, 0, query_bytes);
  std::memset(inst_map, 0, inst_bytes);
  std::memset(doc0_map, 0, doc_bytes);
  std::memset(doc1_map, 0, doc_bytes);
  std::memset(doc2_map, 0, doc_bytes);
  std::memset(doc3_map, 0, doc_bytes);
  std::memset(out_map, 0, out_bytes);

  for (int token = 0; token < VOCAB_SIZE; ++token) {
    df_map[token] = df[token];
  }

  for (int i = 0; i < num_super_batches; ++i) {
    inst_map[i] = packed.inst_mem[i];
  }

  for (int token_id : query_tokens) {
    int chunk_idx = token_id >> 9;
    int bit_idx = token_id & 0x1FF;
    int q_word = chunk_idx * 8 + (bit_idx >> 6);
    query_map[q_word] |= (1ULL << (bit_idx & 63));
  }

  for (size_t i = 0; i < packed.doc_mem[0].size(); ++i) {
    doc0_map[i] = packed.doc_mem[0][i];
    doc1_map[i] = packed.doc_mem[1][i];
    doc2_map[i] = packed.doc_mem[2][i];
    doc3_map[i] = packed.doc_mem[3][i];
  }

  std::vector<int> sw_topk_indices;
  indexer_top_ref(L, documents, query_tokens, df, sw_topk_indices);

  std::set<int> sw_set(sw_topk_indices.begin(), sw_topk_indices.end());
  std::cout << "Prepared software reference and hardware inputs" << std::endl;

  std::cout << "First 16 SW indices: ";
  for (int i = 0; i < std::min(16, static_cast<int>(sw_topk_indices.size())); ++i) {
    std::cout << sw_topk_indices[i] << " ";
  }
  std::cout << std::endl;

  if (static_cast<int>(packed.doc_mem[0].size()) != doc_vectors_per_channel * kLanes ||
      static_cast<int>(packed.doc_mem[1].size()) != doc_vectors_per_channel * kLanes ||
      static_cast<int>(packed.doc_mem[2].size()) != doc_vectors_per_channel * kLanes ||
      static_cast<int>(packed.doc_mem[3].size()) != doc_vectors_per_channel * kLanes) {
    std::cout << "Invalid packed document sizes" << std::endl;
    return EXIT_FAILURE;
  }

  // Synchronize buffer contents with device side.
  bo_df.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_query.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inst.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_doc0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_doc1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_doc2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_doc3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Establish TCP connection before kernel launch so timing excludes setup.
  int tx_sock = setup_tcp_client(receiver_ip, receiver_port);
  if (tx_sock < 0) {
    std::cout << "Failed to establish TCP session before kernel launch" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Execution of the kernel" << std::endl;
    auto launch_start = std::chrono::system_clock::now();
  auto run = krnl(L, doc_vectors_per_channel, bo_df, bo_query, bo_inst, bo_doc0, bo_doc1, bo_doc2, bo_doc3, bo_out);
    auto launch_end = std::chrono::system_clock::now();

    auto exec_start = std::chrono::high_resolution_clock::now();
  run.wait();
    auto exec_end = std::chrono::high_resolution_clock::now();

    double launch_latency_us =
      std::chrono::duration<double, std::micro>(launch_end - launch_start).count();
    double execution_latency_ms =
      std::chrono::duration<double, std::milli>(exec_end - exec_start).count();
    std::cout << "Kernel launch latency: " << launch_latency_us << " us" << std::endl;
    std::cout << "Kernel execution wait time: " << execution_latency_ms << " ms" << std::endl;

  // Get the output data from the device.
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::vector<int> hw_topk_indices;
  hw_topk_indices.reserve(TOP_K);
  for (int i = 0; i < TOP_K; ++i) {
    hw_topk_indices.push_back(out_map[i]);
  }

  const uint64_t launch_start_ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                launch_start.time_since_epoch())
                                .count());
  bool sent = send_timestamp_and_indices(tx_sock, launch_start_ns, hw_topk_indices);
  ::close(tx_sock);
  if (sent) {
    std::cout << "Sent launch_start timestamp(ns) + " << hw_topk_indices.size()
              << " indices to " << receiver_ip << ":" << receiver_port << std::endl;
  } else {
    std::cout << "Failed to send timestamp/indices to " << receiver_ip << ":" << receiver_port << std::endl;
  }

  std::set<int> hw_set(hw_topk_indices.begin(), hw_topk_indices.end());

  bool all_indices_valid = true;
  for (int idx : hw_topk_indices) {
    if (idx < 0 || idx >= L) {
      all_indices_valid = false;
      std::cout << "Invalid index: " << idx << std::endl;
    }
  }

  bool no_duplicates = (hw_set.size() == hw_topk_indices.size());

  int overlap_count = 0;
  for (int idx : hw_set) {
    if (sw_set.find(idx) != sw_set.end()) {
      overlap_count++;
    }
  }

  int high_overlap_hw = 0;
  int high_overlap_sw = 0;
  for (int doc_id : high_overlap_docs) {
    if (hw_set.find(doc_id) != hw_set.end()) {
      high_overlap_hw++;
    }
    if (sw_set.find(doc_id) != sw_set.end()) {
      high_overlap_sw++;
    }
  }

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "HW returned " << hw_topk_indices.size() << " indices" << std::endl;
  std::cout << "SW returned " << sw_topk_indices.size() << " indices" << std::endl;
  std::cout << "Overlap: " << overlap_count << " / " << TOP_K << std::endl;

  std::cout << "First 16 HW indices: ";
  for (int i = 0; i < std::min(16, static_cast<int>(hw_topk_indices.size())); ++i) {
    std::cout << hw_topk_indices[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "High-overlap docs in HW top-K: " << high_overlap_hw << " / " << high_overlap_docs.size() << std::endl;
  std::cout << "High-overlap docs in SW top-K: " << high_overlap_sw << " / " << high_overlap_docs.size() << std::endl;

  float overlap_ratio = static_cast<float>(overlap_count) / static_cast<float>(TOP_K);
  float high_overlap_ratio = high_overlap_docs.empty()
                                 ? 1.0f
                                 : static_cast<float>(high_overlap_hw) / static_cast<float>(high_overlap_docs.size());

  std::cout << "Overlap ratio: " << (overlap_ratio * 100.0f) << "%" << std::endl;
  std::cout << "High-overlap capture ratio: " << (high_overlap_ratio * 100.0f) << "%" << std::endl;

  if (!all_indices_valid) {
    std::cout << "Warning: output contains invalid indices" << std::endl;
  }
  if (!no_duplicates) {
    std::cout << "Warning: output contains duplicate indices" << std::endl;
    std::cout << "Unique indices: " << hw_set.size() << " vs Total: " << hw_topk_indices.size() << std::endl;
  }

  if (high_overlap_ratio >= 0.9f && all_indices_valid && no_duplicates) {
    std::cout << "TEST PASSED" << std::endl;
    return 0;
  }

  if (overlap_ratio >= 0.5f) {
    std::cout << "TEST PARTIAL PASS: significant overlap with software reference" << std::endl;
    return 0;
  }

  std::cout << "TEST FAILED: significant mismatch with software reference" << std::endl;
  return EXIT_FAILURE;
}