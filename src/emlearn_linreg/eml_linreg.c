
#include <math.h>

// ElasticNet implementation (embedded from linreg.c)
typedef struct {
    float* weights;           
    float* weight_gradients;  
    float bias;
    uint16_t n_features;
    float l1_ratio;           
    float alpha;              
    float learning_rate;      
} elastic_net_model_t;

// Soft thresholding function for L1 penalty
static float soft_threshold(float x, float threshold) {
    if (x > threshold) {
        return x - threshold;
    } else if (x < -threshold) {
        return x + threshold;
    } else {
        return 0.0f;
    }
}

// Calculate prediction for a single sample
static float predict_sample(const elastic_net_model_t* model, const float* features) {
    float prediction = model->bias;
    for (uint16_t i = 0; i < model->n_features; i++) {
        prediction += model->weights[i] * features[i];
    }
    return prediction;
}

// Single iteration of gradient descent
static void elastic_net_iterate(elastic_net_model_t* model,
                        const float* X,
                        const float* y,
                        uint16_t n_samples) {
    
    // Initialize gradients buffer to zero
    memset(model->weight_gradients, 0, model->n_features * sizeof(float));
    float bias_gradient = 0.0f;
    
    // Forward pass and gradient calculation
    for (uint16_t i = 0; i < n_samples; i++) {
        // Calculate prediction
        float prediction = predict_sample(model, &X[i * model->n_features]);
        
        // Calculate error
        float error = prediction - y[i];
        
        // Accumulate gradients
        bias_gradient += error;
        for (uint16_t j = 0; j < model->n_features; j++) {
            model->weight_gradients[j] += error * X[i * model->n_features + j];
        }
    }
    
    // Average gradients
    bias_gradient /= n_samples;
    for (uint16_t j = 0; j < model->n_features; j++) {
        model->weight_gradients[j] /= n_samples;
    }
    
    // Update weights with regularization
    for (uint16_t j = 0; j < model->n_features; j++) {
        // Add L2 penalty to gradient
        float l2_penalty = model->alpha * (1.0f - model->l1_ratio) * model->weights[j];
        
        // Update weight
        float new_weight = model->weights[j] - model->learning_rate * (model->weight_gradients[j] + l2_penalty);
        
        // Apply L1 penalty via soft thresholding
        float l1_penalty = model->alpha * model->l1_ratio * model->learning_rate;
        model->weights[j] = soft_threshold(new_weight, l1_penalty);
    }
    
    // Update bias (no regularization on bias)
    model->bias -= model->learning_rate * bias_gradient;
}

// Calculate mean squared error
static float elastic_net_mse(const elastic_net_model_t* model,
                     const float* X,
                     const float* y,
                     uint16_t n_samples) {
    
    float mse = 0.0f;
    for (uint16_t i = 0; i < n_samples; i++) {
        float prediction = predict_sample(model, &X[i * model->n_features]);
        float error = y[i] - prediction;
        mse += error * error;
    }
    return mse / n_samples;
}
