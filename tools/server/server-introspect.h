#pragma once

// PS5LM fork: the introspection ("interpretability") API.
//
// llama.cpp computes, but never exposes, the two things an interpretability
// experiment needs: the raw logit distribution over the whole vocabulary at a
// chosen position, and the intermediate activations of the transformer blocks
// (the residual stream before and after each block, and the MLP/attention
// outputs inside it). This module supplies both, driven by a request field
// ("introspect") rather than by a second model pass:
//
//   - logits are read straight out of the context's output buffer with
//     llama_get_logits_ith() after the existing decode step;
//   - activations are copied out of the compute graph while it runs, through
//     llama_context_params.cb_eval (ggml_backend_sched_eval_callback), which
//     the scheduler calls for every node. The fetch is a no-op unless a
//     request armed it, so ordinary generation pays nothing.
//
// Everything here is request-scoped: the capture engine is armed for one
// decode step at a time and cleared straight after, on the server's own
// (main loop) thread.

#include "server-common.h" // json alias, common_token_to_piece

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

// Default activation tensors, in the order a llama-architecture graph emits
// them (see src/models/llama.cpp):
//   inp_embd            token embedding + position, the residual stream at the input
//   attn_norm-<il>      residual stream after the block's input norm
//   attn_out-<il>       attention output (before the residual add)
//   ffn_norm-<il>       residual stream after the post-attention norm
//   ffn_out-<il>        MLP output (before the residual add)
//   l_out-<il>          residual stream after block il  (l_out-<il-1> is the block input)
//   result_norm         final hidden state before the output projection
//   result_output       logits tensor of the graph
extern const char * const server_introspect_default_tensors[];

// One in-place intervention on an activation tensor: zero, scale or add a value
// to a channel (a single neuron, an attention head dimension, or a range), at one
// position or at every position of the request. Applied inside the graph-eval
// callback, i.e. after the tensor is computed and before its consumers run.
struct server_introspect_intervention {
    std::string          tensor;        // base name ("ffn_out") or exact name ("ffn_out-12")
    std::vector<int32_t> layers;        // layer filter for a base name, empty = all
    std::vector<int64_t> channels;      // explicit channel indices, empty = the range below
    int64_t channel_start = -1;         // inclusive, -1 = from 0
    int64_t channel_end   = -1;         // exclusive, -1 = to the end
    int64_t position      = -1;         // token index of the request, -1 = every position
    std::string op        = "zero";     // zero | scale | add
    float   value         = 0.0f;       // operand of scale/add

    std::vector<std::string> names;     // resolved ggml tensor names
    int64_t modified = 0;               // elements changed, summed over the request

    void build_names(int32_t n_layer);
    bool matches(const std::string & name) const;
    json to_json() const;
};

struct server_introspect_params {
    bool enabled = false;

    // logits
    bool    want_logits   = false;
    int32_t top_k         = 0;     // entries in "top" (0 = no list)
    bool    want_entropy  = false; // one extra pass over the vocabulary
    bool    want_probs    = true;  // probabilities for argmax/top (needs the same pass)
    bool    full          = false; // include the whole float32 row (base64)
    int32_t max_full_rows = 8;     // cap on the number of full rows, bounds the response

    // which positions of the request to look at
    bool                 all_positions = false;   // decode one token at a time so every position is visible
    std::vector<int32_t> positions;               // explicit positions (implies all_positions)

    // activations
    std::vector<std::string> tensors;             // base names, e.g. {"l_out", "ffn_out"}
    std::vector<int32_t>     layers;              // layer suffix filter, empty = all

    bool capture_generated = true;                // also capture the tokens the model produces

    std::vector<server_introspect_intervention> interventions;

    int32_t step_tokens() const {
        if (all_positions) {
            return 1;
        }
        // a position-specific intervention needs every position to be its own row
        for (const auto & iv : interventions) {
            if (iv.position >= 0) {
                return 1;
            }
        }
        return 0;
    }

    bool wants_position(int32_t idx, int32_t n_prompt) const {
        if (!all_positions) {
            return idx == n_prompt - 1; // "last" = the final prompt token
        }
        if (positions.empty()) {
            return true;
        }
        for (const int32_t p : positions) {
            if (p == idx) {
                return true;
            }
        }
        return false;
    }

    bool want_activations() const { return !tensors.empty(); }
    bool want_interventions() const { return !interventions.empty(); }
    bool any() const { return want_logits || want_activations() || want_interventions(); }
};

// Parses (and validates) the "introspect" request field. Throws json_error on a bad field.
server_introspect_params server_introspect_params_from_json(const json & j);
json                    server_introspect_params_to_json(const server_introspect_params & p);

// One tensor row copied out of the graph: ne is the tensor's full shape, data has ne[0] floats.
struct server_introspect_tensor {
    std::string          name;
    std::vector<int64_t> ne;
    std::vector<float>   data;

    json to_json() const;
};

// The graph-eval capture engine. Armed by the slot for one decode step; the
// scheduler then offers it every node of the graph being computed.
struct server_introspect_capture {
    bool armed = false;

    std::unordered_set<std::string> names;  // exact ggml tensor names to match
    int64_t row = -1;                       // row to keep within the tensor, -1 = the last row
    std::vector<int64_t> row_pos;           // sequence position (pos[0]) of each row of the chunk being decoded

    std::vector<server_introspect_intervention> ivs;  // applied when a tensor matches one

    std::map<std::string, server_introspect_tensor> data;  // by tensor name, last write wins

    size_t max_bytes = 256u << 20;  // hard bound on one captured step
    size_t n_bytes   = 0;

    void clear() {
        data.clear();
        n_bytes = 0;
        for (auto & iv : ivs) {
            iv.modified = 0;
        }
    }

    void build_names(const server_introspect_params & p, int32_t n_layer);
    bool on_tensor(struct ggml_tensor * t, bool ask);
};

// the callback installed on the context (never null; a no-op until armed)
bool server_introspect_cb(struct ggml_tensor * t, bool ask, void * user_data);

// Formats the raw logits of one position: argmax, optional top-k / entropy / full row.
// allow_full=false suppresses the full row (used when the per-request cap is reached).
void server_introspect_logits(
        const float *         logits,
        int32_t               n_vocab,
        const llama_vocab *   vocab,
        const server_introspect_params & p,
        bool                  allow_full,
        json &                out);

// Accumulated result of one introspection request, serialized as the response's
// "introspection" object.
struct server_introspect_result {
    json meta      = nullptr;  // model / context facts, filled once by the caller
    json inputs    = nullptr;  // the request's token ids and pieces
    json positions = nullptr;  // one entry per captured position
    json interventions = nullptr; // what was changed, with the count of elements touched
    bool truncated = false;    // the byte budget cut some activations out

    bool empty() const {
        return (positions.is_null() || positions.size() == 0) && interventions.is_null();
    }

    json to_json() const {
        if (empty()) {
            return nullptr;
        }
        json out = json {
            {"meta",      meta},
            {"inputs",    inputs},
            {"positions", positions.is_null() ? json::array() : positions},
            {"truncated", truncated},
        };
        if (!interventions.is_null()) {
            out["interventions"] = interventions;
        }
        return out;
    }
};
