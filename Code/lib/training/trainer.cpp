#include "training/trainer.h"
#include "training/clip_grad.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>

Trainer::Trainer(Sequential& model, Optimizer& optimizer, MSELoss& loss_fn,
                 std::shared_ptr<CosineSchedule> scheduler, float grad_clip_norm)
    : model_(model), optimizer_(optimizer), loss_fn_(loss_fn), scheduler_(scheduler),
      grad_clip_norm_(grad_clip_norm) {
    history["train_loss"] = {};
    history["eval_loss"] = {};
    history["learning_rates"] = {};
}

float Trainer::train_epoch(DataLoader& dataloader, int accumulation_steps) {
    training_mode_ = true;
    float total_loss = 0.0f;
    int num_batches = 0;
    float accumulated_loss = 0.0f;

    // Loop over C++ DataLoader batches
    for (auto batch : dataloader) {
        Tensor& inputs = batch[0];
        Tensor& targets = batch[1];

        // 1. Forward pass
        Tensor outputs = model_.forward(inputs);
        Tensor loss_tensor = loss_fn_.forward(outputs, targets);
        float loss_val = loss_tensor.data()[0];

        // Scale loss for accumulation
        float scaled_loss = loss_val / static_cast<float>(accumulation_steps);
        accumulated_loss += scaled_loss;

        // 2. Backward pass
        loss_tensor.backward();

        // 3. Step and Zero Grad (handled every accumulation_steps)
        if ((num_batches + 1) % accumulation_steps == 0) {
            if (grad_clip_norm_ > 0.0f) {
                clip_grad_norm(model_.parameters(), grad_clip_norm_);
            }
            optimizer_.step();
            optimizer_.zero_grad();

            total_loss += accumulated_loss;
            accumulated_loss = 0.0f;
            step++;
        }
        num_batches++;
    }

    // Handle remaining accumulated gradients if any
    if (accumulated_loss > 0.0f) {
        if (grad_clip_norm_ > 0.0f) {
            clip_grad_norm(model_.parameters(), grad_clip_norm_);
        }
        optimizer_.step();
        optimizer_.zero_grad();
        total_loss += accumulated_loss;
        step++;
    }

    float avg_loss = num_batches > 0 ? (total_loss / (num_batches / accumulation_steps)) : 0.0f;
    history["train_loss"].push_back(avg_loss);

    // Update scheduler
    if (scheduler_) {
        float current_lr = scheduler_->get_lr(epoch);
        // Assuming your AdamW/SGD cast dynamic cast or has a set_lr() interface
        // optimizer_.set_lr(current_lr);
        history["learning_rates"].push_back(current_lr);
    }

    epoch++;
    return avg_loss;
}

std::pair<float, float> Trainer::evaluate(DataLoader& dataloader) {
    training_mode_ = false;
    float total_loss = 0.0f;
    int num_batches = 0;

    for (auto batch : dataloader) {
        Tensor& inputs = batch[0];
        Tensor& targets = batch[1];

        // Forward ONLY (No backward pass, no param updates)
        Tensor outputs = model_.forward(inputs);
        Tensor loss_tensor = loss_fn_.forward(outputs, targets);

        total_loss += loss_tensor.data()[0];
        num_batches++;
    }

    float avg_loss = num_batches > 0 ? (total_loss / num_batches) : 0.0f;
    history["eval_loss"].push_back(avg_loss);

    // For simplicity in XOR, accuracy is omitted, returning 0.0f as placeholder
    return {avg_loss, 0.0f};
}

// Pure C++ Checkpointing (Binary format instead of Python Pickle)
void Trainer::save_checkpoint(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open checkpoint file for writing.");

    out.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));

    auto params = model_.parameters();
    size_t num_params = params.size();
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (Tensor* param : params) {
        size_t param_size = param->size();
        out.write(reinterpret_cast<const char*>(&param_size), sizeof(param_size));
        out.write(reinterpret_cast<const char*>(param->data().data()), param_size * sizeof(float));
    }
    out.close();
}

void Trainer::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open checkpoint file for reading.");

    in.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));
    in.read(reinterpret_cast<char*>(&step), sizeof(step));

    size_t num_params;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    auto params = model_.parameters();
    if (num_params != params.size())
        throw std::runtime_error("Model architecture mismatch in checkpoint.");

    for (Tensor* param : params) {
        size_t param_size;
        in.read(reinterpret_cast<char*>(&param_size), sizeof(param_size));
        if (param_size == param->size()) {
            in.read(reinterpret_cast<char*>(param->data().data()), param_size * sizeof(float));
        }
    }
    in.close();
}
