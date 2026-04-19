#include <math.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>

static float logf_compat(float x) {
    if (x <= 0.0f) {
        x = 1e-6f;
    }
    float y = x;
    int shift = 0;
    while (y > 2.0f) {
        y *= 0.5f;
        shift++;
    }
    while (y < 1.0f) {
        y *= 2.0f;
        shift--;
    }
    float t = y - 1.0f;
    float log_m = t - 0.5f * t * t + (t * t * t) / 3.0f;
    return shift * 0.69314718056f + log_m;
}

static float expf_compat(float x) {
    if (x < -10.0f) {
        x = -10.0f;
    }
    if (x > 10.0f) {
        x = 10.0f;
    }
    const float ln2 = 0.69314718056f;
    int n = (int)(x / ln2);
    float remainder = x - n * ln2;
    float term = 1.0f;
    float result = 1.0f;
    for (int i = 1; i <= 6; i++) {
        term *= remainder / i;
        result += term;
    }
    float two_n = 1.0f;
    if (n >= 0) {
        for (int i = 0; i < n; i++) {
            two_n *= 2.0f;
        }
    } else {
        for (int i = 0; i < -n; i++) {
            two_n *= 0.5f;
        }
    }
    return result * two_n;
}


// Logistic Regression with elastic-net style regularization
// Provides support for simultaneous L1 and L2 penalties.
typedef struct {
    float *weights;
    float *weight_gradients;
    float *biases;
    uint16_t n_features;
    uint16_t n_classes;
    float learning_rate;
    float lambda_l2;
    float lambda_l1;
} logreg_model_t;

typedef struct {
    float *logits;
    float *probabilities;
    float *bias_gradients;
    uint16_t logits_size;
    uint16_t probabilities_size;
    uint16_t bias_gradients_size;
} logreg_workspace_t;

static bool logreg_workspace_validate(const logreg_workspace_t *workspace,
                                      uint16_t required_size,
                                      bool need_bias) {
    if (workspace == NULL) {
        return false;
    }
    if (workspace->logits == NULL || workspace->probabilities == NULL) {
        return false;
    }
    if (workspace->logits_size < required_size ||
        workspace->probabilities_size < required_size) {
        return false;
    }
    if (need_bias) {
        if (workspace->bias_gradients == NULL ||
            workspace->bias_gradients_size < required_size) {
            return false;
        }
    }
    return true;
}

static float soft_threshold(float value, float threshold) {
    if (value > threshold) {
        return value - threshold;
    } else if (value < -threshold) {
        return value + threshold;
    }
    return 0.0f;
}

// logits buffer must be size n_classes
static void logreg_predict_scores(const logreg_model_t *model,
                                  const float *features,
                                  float *scores) {
    const uint16_t n_classes = model->n_classes;
    const uint16_t n_features = model->n_features;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        const float *weights_cls = &model->weights[cls * n_features];
        float logit = model->biases[cls];
        for (uint16_t feat = 0; feat < n_features; feat++) {
            logit += weights_cls[feat] * features[feat];
        }
        scores[cls] = logit;
    }
}

// logits and probabilities buffers must be size n_classes
static void logreg_softmax(const float *logits,
                           uint16_t n_classes,
                           float *probabilities) {
    float max_logit = -FLT_MAX;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        if (logits[cls] > max_logit) {
            max_logit = logits[cls];
        }
    }

    float sum = 0.0f;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        float value = expf_compat(logits[cls] - max_logit);
        probabilities[cls] = value;
        sum += value;
    }

    const float inv_sum = 1.0f / sum;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        probabilities[cls] *= inv_sum;
    }
}

static void logreg_predict_softmax(const logreg_model_t *model,
                                   const float *features,
                                   float *probabilities,
                                   float *logits) {
    const uint16_t n_classes = model->n_classes;
    logreg_predict_scores(model, features, logits);
    logreg_softmax(logits, n_classes, probabilities);
}

bool logreg_iterate(logreg_model_t *model,
                     const float *X,
                     const float *y,
                     uint16_t n_samples,
                     logreg_workspace_t *workspace) {
    if (n_samples == 0 || model->n_classes == 0) {
        return true;
    }

    if (!logreg_workspace_validate(workspace, model->n_classes, true)) {
        return false;
    }

    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;

    memset(model->weight_gradients, 0, n_classes * n_features * sizeof(float));
    memset(workspace->bias_gradients, 0, n_classes * sizeof(float));

    float *logits_ptr = workspace->logits;
    float *probs_ptr = workspace->probabilities;

    for (uint16_t sample = 0; sample < n_samples; sample++) {
        const float *features = &X[sample * n_features];
        const float *target = &y[sample * n_classes];
        logreg_predict_softmax(model, features, probs_ptr, logits_ptr);

        for (uint16_t cls = 0; cls < n_classes; cls++) {
            const float error = probs_ptr[cls] - target[cls];
            workspace->bias_gradients[cls] += error;
            float *grad_weights = &model->weight_gradients[cls * n_features];
            for (uint16_t feat = 0; feat < n_features; feat++) {
                grad_weights[feat] += error * features[feat];
            }
        }
    }

    const float inv_samples = 1.0f / (float)n_samples;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        workspace->bias_gradients[cls] *= inv_samples;
        float *grad_weights = &model->weight_gradients[cls * n_features];
        for (uint16_t feat = 0; feat < n_features; feat++) {
            grad_weights[feat] *= inv_samples;
        }
    }

    const float lr = model->learning_rate;
    const float l2 = model->lambda_l2;
    const float l1 = model->lambda_l1;
    const float l1_threshold = lr * l1;

    for (uint16_t cls = 0; cls < n_classes; cls++) {
        float *weights_cls = &model->weights[cls * n_features];
        float *grad_weights = &model->weight_gradients[cls * n_features];
        for (uint16_t feat = 0; feat < n_features; feat++) {
            float grad = grad_weights[feat] + l2 * weights_cls[feat];
            float updated = weights_cls[feat] - lr * grad;
            weights_cls[feat] = soft_threshold(updated, l1_threshold);
        }
        model->biases[cls] -= lr * workspace->bias_gradients[cls];
    }

    return true;
}

bool logreg_logloss(const logreg_model_t *model,
                     const float *X,
                     const float *y,
                     uint16_t n_samples,
                     logreg_workspace_t *workspace,
                     float *loss_out) {
    if (loss_out == NULL) {
        return false;
    }

    if (n_samples == 0 || model->n_classes == 0) {
        *loss_out = 0.0f;
        return true;
    }

    if (!logreg_workspace_validate(workspace, model->n_classes, false)) {
        *loss_out = 0.0f;
        return false;
    }

    const float eps = 1e-7f;
    float loss = 0.0f;
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;
    float *logits_ptr = workspace->logits;
    float *probs_ptr = workspace->probabilities;

    for (uint16_t sample = 0; sample < n_samples; sample++) {
        const float *features = &X[sample * n_features];
        const float *target = &y[sample * n_classes];
        logreg_predict_softmax(model, features, probs_ptr, logits_ptr);
        for (uint16_t cls = 0; cls < n_classes; cls++) {
            float prediction = probs_ptr[cls];
            if (prediction < eps) {
                prediction = eps;
            } else if (prediction > 1.0f - eps) {
                prediction = 1.0f - eps;
            }
            loss -= target[cls] * logf_compat(prediction);
        }
    }

    loss /= (float)n_samples;

    if (model->lambda_l2 > 0.0f || model->lambda_l1 > 0.0f) {
        float l2_term = 0.0f;
        float l1_term = 0.0f;
        const uint32_t total_weights = (uint32_t)n_features * (uint32_t)n_classes;
        for (uint32_t idx = 0; idx < total_weights; idx++) {
            const float weight = model->weights[idx];
            l2_term += weight * weight;
            l1_term += fabsf(weight);
        }
        loss += 0.5f * model->lambda_l2 * l2_term + model->lambda_l1 * l1_term;
    }

    *loss_out = loss;
    return true;
}
