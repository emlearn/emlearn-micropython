// SEFR: a linear classifier trained in a single pass over the data
// Keshavarz, Abadeh, Rawassizadeh. SEFR: A Fast Linear-Time Classifier
// for Ultra-Low Power Devices. https://arxiv.org/abs/2006.04620
//
// Training keeps only per-class feature sums and the feature ranges,
// so samples can be streamed and discarded. Multiclass uses one-vs-rest.

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define SEFR_EPS 1e-7f

// Natural log of a positive x, without libm (logf does not link in native modules).
// x = m * 2^e with m in [sqrt(0.5), sqrt(2)), then log(m) = 2 atanh((m - 1) / (m + 1))
static float sefr_logf(float x) {
    int e = 0;
    while (x >= 1.41421356f) {
        x *= 0.5f;
        e++;
    }
    while (x < 0.70710678f) {
        x *= 2.0f;
        e--;
    }
    const float s = (x - 1.0f) / (x + 1.0f);
    const float s2 = s * s;
    const float series = 1.0f + s2 * (1.0f / 3 + s2 * (1.0f / 5 + s2 * (1.0f / 7 + s2 * (1.0f / 9))));
    return 2.0f * s * series + (float)e * 0.69314718f;
}

typedef struct {
    float *class_sums;       // n_classes * n_features, sum of each feature per class
    float *feature_min;      // n_features
    float *feature_max;      // n_features
    uint32_t *class_counts;  // n_classes
    float *weights;          // n_classes * n_features, set by sefr_finalize()
    float *biases;           // n_classes, set by sefr_finalize()
    // optional second pass: linear discriminant analysis on the per-class scores
    float *lda_scatter;      // n_classes * n_classes, within-class scatter of the scores
    float *lda_means;        // n_classes * n_classes, mean scores of each class
    float *lda_temp;         // 2 * n_classes
    uint32_t lda_count;
    uint16_t n_features;
    uint16_t n_classes;
} sefr_model_t;

static void sefr_reset(sefr_model_t *model) {
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;
    memset(model->class_sums, 0, sizeof(float) * n_classes * n_features);
    memset(model->class_counts, 0, sizeof(uint32_t) * n_classes);
    memset(model->weights, 0, sizeof(float) * n_classes * n_features);
    memset(model->biases, 0, sizeof(float) * n_classes);
    for (uint16_t feat = 0; feat < n_features; feat++) {
        model->feature_min[feat] = FLT_MAX;
        model->feature_max[feat] = -FLT_MAX;
    }
}

// Add samples to the running statistics. y holds class indices.
// Returns false if a label is out of range.
static bool sefr_partial_fit(sefr_model_t *model,
                             const float *X,
                             const float *y,
                             uint32_t n_samples) {
    const uint16_t n_features = model->n_features;
    for (uint32_t sample = 0; sample < n_samples; sample++) {
        const int32_t label = (int32_t)y[sample];
        if (label < 0 || label >= model->n_classes || (float)label != y[sample]) {
            return false;
        }
        const float *features = &X[sample * n_features];
        float *sums = &model->class_sums[label * n_features];
        for (uint16_t feat = 0; feat < n_features; feat++) {
            const float value = features[feat];
            sums[feat] += value;
            if (value < model->feature_min[feat]) {
                model->feature_min[feat] = value;
            }
            if (value > model->feature_max[feat]) {
                model->feature_max[feat] = value;
            }
        }
        model->class_counts[label] += 1;
    }
    return true;
}

// Compute weights and biases from the statistics.
// Features are min-max scaled to [0, 1], as SEFR needs non-negative inputs,
// and the scaling is folded into the weights and biases,
// so prediction is a plain dot product.
// Returns false unless at least two classes have samples.
static bool sefr_finalize(sefr_model_t *model) {
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;

    uint32_t n_total = 0;
    uint16_t n_present = 0;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        n_total += model->class_counts[cls];
        n_present += (model->class_counts[cls] > 0) ? 1 : 0;
    }
    if (n_present < 2) {
        return false;
    }

    for (uint16_t cls = 0; cls < n_classes; cls++) {
        float *weights = &model->weights[cls * n_features];
        const uint32_t n_pos = model->class_counts[cls];
        const uint32_t n_neg = n_total - n_pos;
        if (n_pos == 0) {
            // a class never seen is never predicted
            memset(weights, 0, sizeof(float) * n_features);
            model->biases[cls] = -FLT_MAX;
            continue;
        }

        float score_pos = 0.0f;
        float score_neg = 0.0f;
        float offset = 0.0f;
        for (uint16_t feat = 0; feat < n_features; feat++) {
            const float range = model->feature_max[feat] - model->feature_min[feat];
            if (!(range > 0.0f)) {
                // constant feature
                weights[feat] = 0.0f;
                continue;
            }
            float total = 0.0f;
            for (uint16_t other = 0; other < n_classes; other++) {
                total += model->class_sums[other * n_features + feat];
            }
            const float sum_pos = model->class_sums[cls * n_features + feat];
            const float min = model->feature_min[feat];
            const float mean_pos = (sum_pos / (float)n_pos - min) / range;
            const float mean_neg = ((total - sum_pos) / (float)n_neg - min) / range;
            const float weight = (mean_pos - mean_neg) / (mean_pos + mean_neg + SEFR_EPS);
            score_pos += weight * mean_pos;
            score_neg += weight * mean_neg;
            // fold the scaling (x - min) / range into the weight and bias
            weights[feat] = weight / range;
            offset += weights[feat] * min;
        }
        // threshold between the class score means, each weighted by the size of the other class
        const float threshold = ((float)n_neg * score_pos + (float)n_pos * score_neg) / (float)n_total;
        model->biases[cls] = -(threshold + offset);
    }
    return true;
}

// Write one score per class, positive meaning the class beats its threshold.
// Returns the class with the highest score.
static uint16_t sefr_predict(const sefr_model_t *model,
                             const float *features,
                             float *scores) {
    const uint16_t n_features = model->n_features;
    uint16_t best_cls = 0;
    for (uint16_t cls = 0; cls < model->n_classes; cls++) {
        const float *weights = &model->weights[cls * n_features];
        float score = model->biases[cls];
        for (uint16_t feat = 0; feat < n_features; feat++) {
            score += weights[feat] * features[feat];
        }
        scores[cls] = score;
        if (score > scores[best_cls]) {
            best_cls = cls;
        }
    }
    return best_cls;
}


// Second pass, step 1: compute the mean scores of each class from the first-pass statistics
// Call after sefr_finalize(). Returns false unless every class has samples.
static bool sefr_lda_begin(sefr_model_t *model) {
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        if (model->class_counts[cls] == 0) {
            return false;
        }
    }
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        const float *sums = &model->class_sums[cls * n_features];
        const float inv_count = 1.0f / (float)model->class_counts[cls];
        for (uint16_t head = 0; head < n_classes; head++) {
            const float *weights = &model->weights[head * n_features];
            float score = model->biases[head];
            for (uint16_t feat = 0; feat < n_features; feat++) {
                score += weights[feat] * sums[feat] * inv_count;
            }
            model->lda_means[cls * n_classes + head] = score;
        }
    }
    memset(model->lda_scatter, 0, sizeof(float) * n_classes * n_classes);
    model->lda_count = 0;
    return true;
}

// Second pass, step 2: accumulate the within-class scatter of the scores
static bool sefr_lda_partial_fit(sefr_model_t *model,
                                 const float *X,
                                 const float *y,
                                 uint32_t n_samples) {
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;
    float *residual = model->lda_temp;
    for (uint32_t sample = 0; sample < n_samples; sample++) {
        const int32_t label = (int32_t)y[sample];
        if (label < 0 || label >= n_classes || (float)label != y[sample]) {
            return false;
        }
        sefr_predict(model, &X[sample * n_features], residual);
        const float *mean = &model->lda_means[label * n_classes];
        for (uint16_t i = 0; i < n_classes; i++) {
            residual[i] -= mean[i];
        }
        for (uint16_t i = 0; i < n_classes; i++) {
            for (uint16_t j = 0; j <= i; j++) {
                model->lda_scatter[i * n_classes + j] += residual[i] * residual[j];
            }
        }
        model->lda_count += 1;
    }
    return true;
}

// Second pass, step 3: replace the scores by the LDA discriminants of the scores.
// Both are linear in the features, so the result is folded into the weights and biases,
// and prediction stays a single dot product per class.
static bool sefr_lda_finalize(sefr_model_t *model) {
    const uint16_t n_features = model->n_features;
    const uint16_t n_classes = model->n_classes;
    const uint16_t K = n_classes;
    float *cov = model->lda_scatter;   // lower triangle, factorized in place
    float *means = model->lda_means;   // row c: class mean scores, replaced by the LDA coefficients
    float *coef = model->lda_temp;
    float *new_bias = &model->lda_temp[K];
    if (model->lda_count == 0) {
        return false;
    }

    // covariance with a small ridge, as the scores of two classes are perfectly correlated
    float trace = 0.0f;
    for (uint16_t i = 0; i < K; i++) {
        for (uint16_t j = 0; j <= i; j++) {
            cov[i * K + j] /= (float)model->lda_count;
        }
        trace += cov[i * K + i];
    }
    const float ridge = 1e-6f * trace / (float)K + 1e-12f;
    for (uint16_t i = 0; i < K; i++) {
        cov[i * K + i] += ridge;
    }

    // Cholesky factorization: cov = L L^T
    for (uint16_t j = 0; j < K; j++) {
        float diag = cov[j * K + j];
        for (uint16_t k = 0; k < j; k++) {
            diag -= cov[j * K + k] * cov[j * K + k];
        }
        if (!(diag > 0.0f)) {
            return false;
        }
        diag = sqrtf(diag);
        cov[j * K + j] = diag;
        for (uint16_t i = j + 1; i < K; i++) {
            float value = cov[i * K + j];
            for (uint16_t k = 0; k < j; k++) {
                value -= cov[i * K + k] * cov[j * K + k];
            }
            cov[i * K + j] = value / diag;
        }
    }

    uint32_t n_total = 0;
    for (uint16_t cls = 0; cls < K; cls++) {
        n_total += model->class_counts[cls];
    }

    for (uint16_t cls = 0; cls < K; cls++) {
        float *mean = &means[cls * K];
        // coef = cov^-1 mean, by forward then backward substitution
        for (uint16_t i = 0; i < K; i++) {
            float value = mean[i];
            for (uint16_t k = 0; k < i; k++) {
                value -= cov[i * K + k] * coef[k];
            }
            coef[i] = value / cov[i * K + i];
        }
        for (int32_t i = K - 1; i >= 0; i--) {
            float value = coef[i];
            for (uint16_t k = i + 1; k < K; k++) {
                value -= cov[k * K + i] * coef[k];
            }
            coef[i] = value / cov[i * K + i];
        }
        // discriminant: coef . score - coef . mean / 2 + log(prior)
        float bias = sefr_logf((float)model->class_counts[cls] / (float)n_total);
        for (uint16_t i = 0; i < K; i++) {
            bias += coef[i] * (model->biases[i] - 0.5f * mean[i]);
        }
        new_bias[cls] = bias;
        memcpy(mean, coef, sizeof(float) * K);
    }

    // fold the coefficients into the weights, one feature column at a time
    for (uint16_t feat = 0; feat < n_features; feat++) {
        for (uint16_t cls = 0; cls < K; cls++) {
            float value = 0.0f;
            for (uint16_t i = 0; i < K; i++) {
                value += means[cls * K + i] * model->weights[i * n_features + feat];
            }
            coef[cls] = value;
        }
        for (uint16_t cls = 0; cls < K; cls++) {
            model->weights[cls * n_features + feat] = coef[cls];
        }
    }
    memcpy(model->biases, new_bias, sizeof(float) * K);
    return true;
}
