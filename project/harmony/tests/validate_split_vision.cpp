// Run actual split-vision MNN files on CPU against an independently saved HF embedding.
// Usage: validate_split_vision model_dir reference_dir chunk_count
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

using namespace MNN;
using namespace MNN::Express;

static VARP bridge(VARP value) {
    if (!value.get()) return nullptr;
    const auto* data = value->readMap<float>();
    return data ? _Const(data, value->getInfo()->dim, value->getInfo()->order, value->getInfo()->type) : nullptr;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string modelDir(argv[1]), referenceDir(argv[2]);
    std::ifstream description(referenceDir + "/inputs.txt");
    if (!description) return 3;
    std::map<std::string, VARP> variables;
    std::string name, dtype;
    int rank;
    while (description >> name >> dtype >> rank) {
        std::vector<int> shape(rank);
        for (auto& dimension : shape) description >> dimension;
        auto value = _Input(shape, NCHW, dtype == "int32" ? halide_type_of<int>() : halide_type_of<float>());
        std::ifstream source(referenceDir + "/" + name + ".bin", std::ios::binary);
        if (!source.read(static_cast<char*>(value->writeMap<void>()), value->getInfo()->size * 4)) return 4;
        variables[name] = value;
    }
    BackendConfig backend;
    backend.precision = BackendConfig::Precision_High;
    ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.numThread = 4;
    schedule.backendConfig = &backend;
    std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(schedule));
    Module::Config config;
    auto load = [&](const std::string& name) {
        runtime->setExternalFile(modelDir + "/" + name + ".mnn.weight");
        return std::unique_ptr<Module>(Module::load({}, {}, (modelDir + "/" + name + ".mnn").c_str(),
                                                   runtime, &config));
    };
    auto pre = load("visual_pre");
    if (!pre) return 5;
    VARPS inputs;
    for (const auto& key : pre->getInfo()->inputNames) {
        if (!variables.count(key)) return 6;
        inputs.push_back(variables[key]);
    }
    auto outputs = pre->onForward(inputs);
    if (outputs.size() != 2) return 7;
    auto hidden = bridge(outputs[0]), rotary = bridge(outputs[1]);
    VARPS deepstack;
    for (int index = 0; index < std::stoi(argv[3]); ++index) {
        auto block = load("visual_blocks_npu_" + std::to_string(index));
        if (!block || !hidden.get() || !rotary.get()) return 8;
        auto values = block->onForward({hidden, rotary, variables.at("attention_mask")});
        if (values.empty()) return 9;
        hidden = bridge(values[0]);
        for (size_t output = 1; output < values.size(); ++output) deepstack.push_back(bridge(values[output]));
    }
    auto post = load("visual_post");
    if (!post) return 10;
    inputs = {hidden};
    inputs.insert(inputs.end(), deepstack.begin(), deepstack.end());
    if (post->getInfo()->inputNames.size() != inputs.size()) return 11;
    auto result = post->onForward(inputs);
    if (result.empty() || !result[0].get()) return 12;
    const int count = result[0]->getInfo()->size;
    const float* actual = result[0]->readMap<float>();
    std::vector<float> expected(count);
    std::ifstream source(referenceDir + "/expected.bin", std::ios::binary);
    if (!actual || !source.read(reinterpret_cast<char*>(expected.data()), count * sizeof(float))) return 13;
    double squaredError = 0.0, normReference = 0.0, normActual = 0.0, dot = 0.0;
    for (int index = 0; index < count; ++index) {
        const double value = actual[index], reference = expected[index];
        if (!std::isfinite(value)) return 14;
        squaredError += (value - reference) * (value - reference);
        normReference += reference * reference;
        normActual += value * value;
        dot += value * reference;
    }
    const double relative = std::sqrt(squaredError / normReference);
    const double cosine = dot / std::sqrt(normReference * normActual);
    std::cout << "VISION elements=" << count << " relative_l2=" << relative << " cosine=" << cosine
              << " deepstack=" << deepstack.size() << "\n";
    // Weight-only W8 conversion must preserve the final image representation.
    return relative < 0.05 && cosine > 0.999 ? 0 : 1;
}
