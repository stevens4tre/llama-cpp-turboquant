#include "server-introspect.h"

#include "common.h"
#include "log.h"

#include "base64.hpp"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

const char * const server_introspect_default_tensors[] = {
    "l_out",
    "ffn_out",
    "attn_out",
    "result_norm",
    nullptr,
};

//
// params
//

static int32_t introspect_int_field(const json & j, const char * key, int32_t def, int32_t lo, int32_t hi) {
    if (!j.contains(key) || j.at(key).is_null()) {
        return def;
    }
    const int32_t v = j.at(key).get<int>();
    if (v < lo || v > hi) {
        throw std::runtime_error(std::string("introspect: '") + key + "' out of range");
    }
    return v;
}

static int64_t introspect_int_field64(const json & j, const char * key, int64_t def, int64_t lo, int64_t hi) {
    if (!j.contains(key) || j.at(key).is_null()) {
        return def;
    }
    const int64_t v = j.at(key).get<long long>();
    if (v < lo || v > hi) {
        throw std::runtime_error(std::string("introspect: '") + key + "' out of range");
    }
    return v;
}

server_introspect_params server_introspect_params_from_json(const json & j) {
    server_introspect_params p;

    if (j.is_null()) {
        return p;
    }
    if (!j.is_object()) {
        throw std::runtime_error("introspect: expected an object");
    }

    // "logits": true | {top_k, entropy, probs, full, max_full_rows}
    if (j.contains("logits") && !j.at("logits").is_null()) {
        const json & l = j.at("logits");
        p.want_logits = true;
        if (l.is_object()) {
            p.top_k         = introspect_int_field(l, "top_k",         20, 0, 65536);
            p.max_full_rows = introspect_int_field(l, "max_full_rows",  8, 0, 4096);
            p.want_entropy  = l.value("entropy", true);
            p.want_probs    = l.value("probs",   true);
            p.full          = l.value("full",    false);
        } else {
            p.top_k = 20;
        }
    }

    // "positions": "last" | "all" | [i, j, ...]
    if (j.contains("positions") && !j.at("positions").is_null()) {
        const json & pos = j.at("positions");
        if (pos.is_string()) {
            const std::string mode = pos.get<std::string>();
            if (mode == "all") {
                p.all_positions = true;
            } else if (mode != "last") {
                throw std::runtime_error("introspect: 'positions' must be \"last\", \"all\" or a list");
            }
        } else if (pos.is_array()) {
            for (size_t i = 0; i < pos.size(); ++i) {
                const int32_t v = pos.at(i).get<int>();
                if (v < 0) {
                    throw std::runtime_error("introspect: 'positions' entries must be >= 0");
                }
                p.positions.push_back(v);
            }
            // every position has to be visible, so the batch is rendered one token at a time
            p.all_positions = true;
        } else {
            throw std::runtime_error("introspect: 'positions' must be a string or a list");
        }
    }

    // "activations": ["l_out", ...] | {"tensors": [...], "layers": "all" | [i, ...]}
    if (j.contains("activations") && !j.at("activations").is_null()) {
        const json & a = j.at("activations");
        const json * t = nullptr;
        const json * l = nullptr;

        if (a.is_object()) {
            if (a.contains("tensors")) {
                t = &a.at("tensors");
            }
            if (a.contains("layers")) {
                l = &a.at("layers");
            }
        } else if (a.is_array()) {
            t = &a;
        } else {
            throw std::runtime_error("introspect: 'activations' must be a list or an object");
        }

        if (t != nullptr) {
            if (!t->is_array()) {
                throw std::runtime_error("introspect: 'activations' tensors must be a list");
            }
            for (size_t i = 0; i < t->size(); ++i) {
                p.tensors.push_back(t->at(i).get<std::string>());
            }
        } else {
            // bare {"layers": [...]} is a shorthand for the default tensors
            for (size_t i = 0; server_introspect_default_tensors[i] != nullptr; ++i) {
                p.tensors.push_back(server_introspect_default_tensors[i]);
            }
        }

        if (l != nullptr && !l->is_null()) {
            if (l->is_string()) {
                if (l->get<std::string>() != "all") {
                    throw std::runtime_error("introspect: 'layers' must be \"all\" or a list");
                }
            } else if (l->is_array()) {
                for (size_t i = 0; i < l->size(); ++i) {
                    const int32_t v = l->at(i).get<int>();
                    if (v < 0) {
                        throw std::runtime_error("introspect: 'layers' entries must be >= 0");
                    }
                    p.layers.push_back(v);
                }
            } else {
                throw std::runtime_error("introspect: 'layers' must be a string or a list");
            }
        }
    }

    if (j.contains("generated") && !j.at("generated").is_null()) {
        p.capture_generated = j.at("generated").get<bool>();
    }

    // "interventions": [{"tensor": "ffn_out", "layers": [12], "channels": [3456], "op": "zero"}]
    if (j.contains("interventions") && !j.at("interventions").is_null()) {
        const json & arr = j.at("interventions");
        if (!arr.is_array()) {
            throw std::runtime_error("introspect: 'interventions' must be a list");
        }
        for (size_t i = 0; i < arr.size(); ++i) {
            const json & e = arr.at(i);
            if (!e.is_object()) {
                throw std::runtime_error("introspect: each intervention must be an object");
            }
            server_introspect_intervention iv;
            if (e.contains("tensor") && !e.at("tensor").is_null()) {
                iv.tensor = e.at("tensor").get<std::string>();
            }
            if (iv.tensor.empty()) {
                throw std::runtime_error("introspect: an intervention needs a 'tensor'");
            }
            if (e.contains("layers") && e.at("layers").is_array()) {
                for (size_t k = 0; k < e.at("layers").size(); ++k) {
                    iv.layers.push_back(e.at("layers").at(k).get<int>());
                }
            }
            if (e.contains("channels") && e.at("channels").is_array()) {
                for (size_t k = 0; k < e.at("channels").size(); ++k) {
                    iv.channels.push_back(e.at("channels").at(k).get<long long>());
                }
            }
            iv.channel_start = introspect_int_field64(e, "channel_start", -1, -1, 1 << 30);
            iv.channel_end   = introspect_int_field64(e, "channel_end",   -1, -1, 1 << 30);
            iv.position      = introspect_int_field64(e, "position",      -1, -1, 1 << 30);
            if (e.contains("op") && !e.at("op").is_null()) {
                iv.op = e.at("op").get<std::string>();
            }
            if (e.contains("value") && !e.at("value").is_null()) {
                iv.value = e.at("value").get<float>();
            }
            if (iv.op != "zero" && iv.op != "scale" && iv.op != "add") {
                throw std::runtime_error("introspect: intervention op must be \"zero\", \"scale\" or \"add\"");
            }
            if (iv.channels.empty() && iv.channel_start < 0 && iv.channel_end < 0) {
                // no channel named: the whole tensor row is the target
                iv.channel_start = 0;
            }
            p.interventions.push_back(std::move(iv));
        }
    }

    p.enabled = p.any();
    return p;
}

json server_introspect_params_to_json(const server_introspect_params & p) {
    json j = json::object();
    if (p.want_logits) {
        j["top_k"]         = p.top_k;
        j["entropy"]       = p.want_entropy;
        j["probs"]         = p.want_probs;
        j["full"]          = p.full;
        j["max_full_rows"] = p.max_full_rows;
    }
    if (p.all_positions) {
        if (p.positions.empty()) {
            j["positions"] = "all";
        } else {
            json pos = json::array();
            for (const int32_t v : p.positions) {
                pos.push_back(v);
            }
            j["positions"] = pos;
        }
    } else {
        j["positions"] = "last";
    }
    if (!p.tensors.empty()) {
        json tensors = json::array();
        for (const auto & t : p.tensors) {
            tensors.push_back(t);
        }
        j["activations"] = json {
            {"tensors", tensors},
            {"layers",  p.layers.empty() ? json("all") : json(p.layers)},
        };
    }
    if (!p.interventions.empty()) {
        json arr = json::array();
        for (const auto & iv : p.interventions) {
            arr.push_back(iv.to_json());
        }
        j["interventions"] = arr;
    }
    j["generated"] = p.capture_generated;
    return j;
}

//
// capture
//

void server_introspect_intervention::build_names(int32_t n_layer) {
    names.clear();
    if (tensor.find('-') != std::string::npos) {
        names.push_back(tensor); // an exact name, e.g. "ffn_out-12"
        return;
    }
    if (layers.empty()) {
        for (int32_t il = 0; il <= n_layer; ++il) {
            names.push_back(tensor + "-" + std::to_string(il));
        }
    } else {
        for (const int32_t il : layers) {
            names.push_back(tensor + "-" + std::to_string(il));
        }
    }
    names.push_back(tensor); // layer-less tensors (inp_embd, result_norm)
}

bool server_introspect_intervention::matches(const std::string & name) const {
    for (const auto & n : names) {
        if (n == name) {
            return true;
        }
    }
    return false;
}

json server_introspect_intervention::to_json() const {
    json channels = json::array();
    for (const int64_t c : this->channels) {
        channels.push_back(c);
    }
    return json {
        {"tensor",        tensor},
        {"layers",        layers.empty() ? json("all") : json(layers)},
        {"channels",      channels},
        {"channel_start", channel_start},
        {"channel_end",   channel_end},
        {"position",      position},
        {"op",            op},
        {"value",         value},
        {"modified",      modified},
    };
}

void server_introspect_capture::build_names(const server_introspect_params & p, int32_t n_layer) {
    names.clear();
    for (const auto & base : p.tensors) {
        // a name that already carries a layer suffix is matched as given
        if (base.find('-') != std::string::npos) {
            names.insert(base);
            continue;
        }
        if (p.layers.empty()) {
            for (int32_t il = 0; il <= n_layer; ++il) {
                names.insert(base + "-" + std::to_string(il));
            }
        } else {
            for (const int32_t il : p.layers) {
                names.insert(base + "-" + std::to_string(il));
            }
        }
        // layer-less tensors (inp_embd, result_norm, result_output) carry no suffix
        names.insert(base);
    }

    ivs = p.interventions;
    for (auto & iv : ivs) {
        iv.build_names(n_layer);
    }
}

bool server_introspect_capture::on_tensor(struct ggml_tensor * t, bool ask) {
    if (!armed) {
        return false;
    }
    const bool wanted = names.find(t->name) != names.end();

    std::vector<server_introspect_intervention *> hits;
    for (auto & iv : ivs) {
        if (iv.matches(t->name)) {
            hits.push_back(&iv);
        }
    }
    if (!wanted && hits.empty()) {
        return false;
    }
    if (ask) {
        // tell the scheduler we want this node: for a device buffer that makes it
        // the end of a split and copies the result back to the host. Everything
        // below runs with the tensor complete and before its consumers run, which
        // is what makes an in-place intervention safe.
        return true;
    }

    // !ask: the node's buffer is complete and host-visible
    const int64_t n_rows = t->ne[1];
    if (n_rows <= 0 || t->ne[0] <= 0) {
        return true;
    }

    const bool want_row = wanted && (row < 0 || row < n_rows) && n_bytes + ggml_row_size(t->type, t->ne[0]) <= max_bytes;
    if (want_row) {
        const int64_t r = row < 0 ? n_rows - 1 : row;
        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        std::vector<uint8_t> raw(row_bytes);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(raw.data(), (const uint8_t *) t->data + (size_t) r * t->nb[1], row_bytes);
        } else {
            ggml_backend_tensor_get(t, raw.data(), (size_t) r * t->nb[1], row_bytes);
        }

        server_introspect_tensor out;
        out.name = t->name;
        out.ne.assign(t->ne, t->ne + GGML_MAX_DIMS);
        out.data.resize(t->ne[0]);

        const int64_t n = t->ne[0];
        bool ok = true;
        switch (t->type) {
            case GGML_TYPE_F32:
                std::memcpy(out.data.data(), raw.data(), (size_t) n * sizeof(float));
                break;
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data.data(), n);
                break;
            case GGML_TYPE_BF16:
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.data.data(), n);
                break;
            default:
                ok = false;
                break;
        }
        if (ok) {
            n_bytes += row_bytes;
            data[t->name] = std::move(out); // the last write wins, i.e. the newest sub-batch
        }
    }

    if (hits.empty()) {
        return true;
    }

    // interventions: read the tensor (or the rows they touch), change it, write it back
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes > (64u << 20)) {
        return true; // refuse to buffer a tensor this large
    }
    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const bool direct = host && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->nb[1] == (size_t) ne0 * sizeof(float);
    if (!direct && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return true; // not a float activation
    }

    std::vector<uint8_t> raw(nbytes);
    if (host) {
        std::memcpy(raw.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }

    // a layout with unit strides in dim 0/1 (how llama.cpp's graph tensors are laid out)
    std::vector<float> v;
    float * dense = nullptr; // mutable pointer into raw when the layout is already unit-stride
    if (direct) {
        dense = reinterpret_cast<float *>(raw.data());
    } else {
        v.resize((size_t) ne0 * ne1 * ne2);
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(v.data(), raw.data(), (size_t) ne0 * ne1 * ne2 * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), v.data(), (int64_t) ne0 * ne1 * ne2);
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), v.data(), (int64_t) ne0 * ne1 * ne2);
        }
        dense = v.data();
    }

    bool changed = false;
    for (auto * iv : hits) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            if (iv->position >= 0 && (i1 >= (int64_t) row_pos.size() || iv->position != row_pos[i1])) {
                continue;
            }
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                bool hit = false;
                if (!iv->channels.empty()) {
                    for (const int64_t c : iv->channels) {
                        if (c == i0) {
                            hit = true;
                            break;
                        }
                    }
                } else {
                    hit = i0 >= (iv->channel_start < 0 ? 0 : iv->channel_start) &&
                          (iv->channel_end < 0 || i0 < iv->channel_end);
                }
                if (!hit) {
                    continue;
                }
                const size_t idx = (size_t) i0 + (size_t) i1 * ne0;
                float * slot_v = dense + idx;
                if (iv->op == "zero") {
                    *slot_v = 0.0f;
                } else if (iv->op == "scale") {
                    *slot_v *= iv->value;
                } else {
                    *slot_v += iv->value;
                }
                iv->modified++;
                changed = true;
            }
        }
    }

    if (changed) {
        if (direct) {
            // dense points into raw, write it back as-is
        } else if (t->type == GGML_TYPE_F32) {
            std::memcpy(raw.data(), v.data(), v.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            for (size_t i = 0; i < v.size(); ++i) {
                ((ggml_fp16_t *) raw.data())[i] = ggml_fp32_to_fp16(v[i]);
            }
        } else {
            for (size_t i = 0; i < v.size(); ++i) {
                ((ggml_bf16_t *) raw.data())[i] = ggml_fp32_to_bf16(v[i]);
            }
        }
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(t->data, raw.data(), nbytes);
        } else {
            ggml_backend_tensor_set(t, raw.data(), 0, nbytes);
        }
    }

    return true;
}

bool server_introspect_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * capture = (server_introspect_capture *) user_data;
    if (capture == nullptr) {
        return false;
    }
    return capture->on_tensor(t, ask);
}

json server_introspect_tensor::to_json() const {
    json ne = json::array();
    for (const int64_t d : this->ne) {
        ne.push_back(d);
    }
    return json {
        {"shape", ne},
        {"dtype", "f32"},
        {"encoding", "base64"},
        {"data", base64::encode((const char *) data.data(), data.size() * sizeof(float))},
    };
}

//
// logits
//

// bounded top-k selection, no allocation proportional to the vocabulary
static std::vector<int32_t> introspect_top_k(const float * logits, int32_t n_vocab, int32_t k) {
    std::vector<int32_t> best;
    if (k <= 0) {
        return best;
    }
    best.reserve(k);

    auto worse = [&](int32_t a, int32_t b) { return logits[a] > logits[b]; }; // min-heap on the logit

    for (int32_t i = 0; i < n_vocab; ++i) {
        if ((int32_t) best.size() < k) {
            best.push_back(i);
            std::push_heap(best.begin(), best.end(), worse);
        } else if (logits[i] > logits[best.front()]) {
            std::pop_heap(best.begin(), best.end(), worse);
            best.back() = i;
            std::push_heap(best.begin(), best.end(), worse);
        }
    }
    std::sort(best.begin(), best.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    return best;
}

static json introspect_token_json(const llama_vocab * vocab, int32_t id, float logit, float prob, bool with_prob) {
    json j = json {
        {"id",    id},
        {"piece", common_token_to_piece(vocab, id, true)},
        {"logit", logit},
    };
    if (with_prob) {
        j["prob"] = prob;
    }
    return j;
}

void server_introspect_logits(
        const float *         logits,
        int32_t               n_vocab,
        const llama_vocab *   vocab,
        const server_introspect_params & p,
        bool                  allow_full,
        json &                out) {
    if (logits == nullptr || n_vocab <= 0) {
        out = nullptr;
        return;
    }

    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    const float best_logit = logits[best];

    const bool need_norm = p.want_probs || p.want_entropy;
    double log_sum = 0.0;
    if (need_norm) {
        double sum = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            sum += std::exp((double) (logits[i] - best_logit));
        }
        log_sum = std::log(sum);
    }
    auto prob_of = [&](float logit) -> float {
        return (float) std::exp((double) (logit - best_logit) - log_sum);
    };

    out = json::object();
    out["n_vocab"] = n_vocab;
    // the argmax's own probability is exp(0 - logsumexp), not 1.0
    out["argmax"]  = introspect_token_json(vocab, best, best_logit,
                                           need_norm ? (float) std::exp(-log_sum) : 0.0f, p.want_probs);

    if (p.top_k > 0) {
        const std::vector<int32_t> top = introspect_top_k(logits, n_vocab, p.top_k);
        json arr = json::array();
        for (const int32_t id : top) {
            arr.push_back(introspect_token_json(vocab, id, logits[id], p.want_probs ? prob_of(logits[id]) : 0.0f, p.want_probs));
        }
        out["top"] = arr;
    }

    if (p.want_entropy) {
        double h = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            const double lp = (double) (logits[i] - best_logit) - log_sum;
            const double pr = std::exp(lp);
            if (pr > 0.0) {
                h -= pr * lp;
            }
        }
        out["entropy"] = h;
    }

    if (p.full && allow_full) {
        out["full"] = json {
            {"encoding", "base64"},
            {"dtype",    "f32"},
            {"n_vocab",  n_vocab},
            {"data",     base64::encode((const char *) logits, (size_t) n_vocab * sizeof(float))},
        };
    }
}
