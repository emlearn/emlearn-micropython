#include <math.h>
#include <string.h>

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
    float bias;
    uint16_t n_features;
    float learning_rate;
    float lambda_l2;
    float lambda_l1;
} logreg_model_t;

static float soft_threshold(float value, float threshold) {
    if (value > threshold) {
        return value - threshold;
    } else if (value < -threshold) {
        return value + threshold;
    }
    return 0.0f;
}


static float sigmoidf(float x) {
    if (x > 10.0f) {
        x = 10.0f;
    } else if (x < -10.0f) {
        x = -10.0f;
    }
    float ex = expf_compat(x);
    return ex / (1.0f + ex);
}

static float logreg_predict_proba(const logreg_model_t *model, const float *features) {
    float logit = model->bias;
    for (uint16_t i = 0; i < model->n_features; i++) {
        logit += model->weights[i] * features[i];
    }
    return sigmoidf(logit);
}

static void logreg_iterate(logreg_model_t *model,
                           const float *X,
                           const float *y,
                           uint16_t n_samples) {
    if (n_samples == 0) {
        return;
    }

    const uint16_t n_features = model->n_features;

    memset(model->weight_gradients, 0, n_features * sizeof(float));
    float bias_gradient = 0.0f;

    for (uint16_t sample = 0; sample < n_samples; sample++) {
        const float *features = &X[sample * n_features];
        const float target = y[sample];
        const float prediction = logreg_predict_proba(model, features);
        const float error = prediction - target;

        bias_gradient += error;
        for (uint16_t feat = 0; feat < n_features; feat++) {
            model->weight_gradients[feat] += error * features[feat];
        }
    }

    const float inv_samples = 1.0f / (float)n_samples;
    bias_gradient *= inv_samples;
    for (uint16_t feat = 0; feat < n_features; feat++) {
        model->weight_gradients[feat] *= inv_samples;
    }

    const float lr = model->learning_rate;
    const float l2 = model->lambda_l2;
    const float l1 = model->lambda_l1;
    const float l1_threshold = lr * l1;

    for (uint16_t feat = 0; feat < n_features; feat++) {
        float grad = model->weight_gradients[feat] + l2 * model->weights[feat];
        float updated = model->weights[feat] - lr * grad;
        model->weights[feat] = soft_threshold(updated, l1_threshold);
    }

    model->bias -= lr * bias_gradient;
}

static float logreg_logloss(const logreg_model_t *model,
                            const float *X,
                            const float *y,
                            uint16_t n_samples) {
    if (n_samples == 0) {
        return 0.0f;
    }

    const float eps = 1e-7f;
    float loss = 0.0f;

    for (uint16_t sample = 0; sample < n_samples; sample++) {
        const float *features = &X[sample * model->n_features];
        float prediction = logreg_predict_proba(model, features);
        if (prediction < eps) {
            prediction = eps;
        } else if (prediction > 1.0f - eps) {
            prediction = 1.0f - eps;
        }
        const float target = y[sample];
        loss -= target * logf_compat(prediction) + (1.0f - target) * logf_compat(1.0f - prediction);
    }

    loss /= (float)n_samples;

    if (model->lambda_l2 > 0.0f || model->lambda_l1 > 0.0f) {
        float l2_term = 0.0f;
        float l1_term = 0.0f;
        for (uint16_t feat = 0; feat < model->n_features; feat++) {
            const float weight = model->weights[feat];
            l2_term += weight * weight;
            l1_term += fabsf(weight);
        }
        loss += 0.5f * model->lambda_l2 * l2_term + model->lambda_l1 * l1_term;
    }

    return loss;
}
