/**
 * @file eml_plsr.h
 * @brief Embedded Machine Learning - Partial Least Squares Regression (NIPALS)
 * 
 * Header-only implementation using single precision floats and a single
 * pre-allocated memory block.
 * 
 * Based on: Wold, H. (1966) "Estimation of principal components and related 
 * models by iterative least squares"
 * 
 * Usage:
 *   // Default: implementation is included (header-only mode)
 *   #include "eml_plsr.h"
 *   
 *   // To disable implementation in some files:
 *   #define EML_PLSR_IMPLEMENTATION 0
 *   #include "eml_plsr.h"
 */

#ifndef EML_PLSR_H
#define EML_PLSR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Calculate required memory size for PLS regression
 * 
 * @param n_samples Number of training samples
 * @param n_features Number of input features
 * @param n_components Number of PLS components
 * @return Required memory size in bytes
 */
#define EML_PLSR_MEMORY_SIZE(n_samples, n_features, n_components) \
    (((size_t)(n_samples) * (size_t)(n_features) + /* inputs_work */ \
      (size_t)(n_samples) +                         /* targets_work */ \
      (size_t)(n_features) * (size_t)(n_components) + /* weights */ \
      (size_t)(n_features) * (size_t)(n_components) + /* loadings_x */ \
      (size_t)(n_components) +                      /* loadings_y */ \
      (size_t)(n_samples) * (size_t)(n_components) + /* scores */ \
      (size_t)(n_samples) +                         /* score_curr */ \
      (size_t)(n_samples) +                         /* score_y_curr */ \
      (size_t)(n_features) +                        /* weight_curr */ \
      (size_t)(n_features)                          /* loading_x_curr */ \
     ) * sizeof(float))

/* Error codes */
typedef enum _EmlError {
    EmlOk = 0,
    EmlSizeMismatch,
    EmlUnsupported,
    EmlUninitialized,
    EmlPostconditionFailed,
    EmlUnknownError,
    EmlErrors,
} EmlError;

/**
 * @brief PLS regression state structure
 */
typedef struct {
    /* Problem dimensions */
    uint16_t n_samples;
    uint16_t n_features;
    uint16_t n_components;
    
    /* Pointers into user-provided memory */
    float *inputs_work;
    float *targets_work;
    float *weights;
    float *loadings_x;
    float *loadings_y;
    float *scores;
    float *score_curr;
    float *score_y_curr;
    float *weight_curr;
    float *loading_x_curr;
    
    /* Training state */
    uint16_t current_component;
    uint16_t current_iter;
    bool component_converged;
    float convergence_metric;
} eml_plsr_t;

/* ============================================================================
 * IMPLEMENTATION CONTROL
 * ============================================================================
 */

#ifndef EML_PLSR_IMPLEMENTATION
#define EML_PLSR_IMPLEMENTATION 1
#endif

#if !EML_PLSR_IMPLEMENTATION
/* Function declarations (only when implementation is disabled) */

EmlError eml_plsr_init(
    eml_plsr_t *plsr,
    uint16_t n_samples,
    uint16_t n_features,
    uint16_t n_components,
    void *memory,
    size_t memory_size
);

size_t eml_plsr_get_memory_size(
    uint16_t n_samples,
    uint16_t n_features,
    uint16_t n_components
);

EmlError eml_plsr_fit_start(
    eml_plsr_t *plsr,
    const float *X,
    const float *y
);

EmlError eml_plsr_iteration_step(
    eml_plsr_t *plsr,
    float tolerance
);

bool eml_plsr_is_converged(const eml_plsr_t *plsr);

EmlError eml_plsr_finalize_component(eml_plsr_t *plsr);

bool eml_plsr_is_complete(const eml_plsr_t *plsr);

EmlError eml_plsr_predict(
    const eml_plsr_t *plsr,
    const float *x,
    float *y_pred
);

EmlError eml_plsr_fit(
    eml_plsr_t *plsr,
    const float *X,
    const float *y,
    uint16_t max_iter,
    float tolerance
);

#endif /* !EML_PLSR_IMPLEMENTATION */

/* ============================================================================
 * IMPLEMENTATION
 * ============================================================================
 */

#if EML_PLSR_IMPLEMENTATION

#include <string.h>
#include <math.h>

/* Internal helper functions - all prefixed with eml_plsr_ */

static inline float eml_plsr_dot_product(const float *a, const float *b, uint16_t n) {
    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float eml_plsr_vector_norm(const float *v, uint16_t n) {
    return sqrtf(eml_plsr_dot_product(v, v, n));
}

static inline float eml_plsr_normalize_vector(float *v, uint16_t n) {
    float norm = eml_plsr_vector_norm(v, n);
    if (norm > 1e-10f) {
        for (uint16_t i = 0; i < n; i++) {
            v[i] /= norm;
        }
    }
    return norm;
}

static inline void eml_plsr_mat_vec_mult(const float *A, const float *x, float *y, 
                                          uint16_t n, uint16_t m) {
    for (uint16_t i = 0; i < n; i++) {
        y[i] = 0.0f;
        for (uint16_t j = 0; j < m; j++) {
            y[i] += A[i * m + j] * x[j];
        }
    }
}

static inline void eml_plsr_mat_trans_vec_mult(const float *A, const float *x, float *y,
                                                 uint16_t n, uint16_t m) {
    for (uint16_t j = 0; j < m; j++) {
        y[j] = 0.0f;
        for (uint16_t i = 0; i < n; i++) {
            y[j] += A[i * m + j] * x[i];
        }
    }
}

static inline void eml_plsr_deflate_matrix(float *A, const float *v, const float *w,
                                            float alpha, uint16_t n, uint16_t m) {
    for (uint16_t i = 0; i < n; i++) {
        for (uint16_t j = 0; j < m; j++) {
            A[i * m + j] -= alpha * v[i] * w[j];
        }
    }
}

static inline void eml_plsr_deflate_vector(float *v, const float *u, float alpha, uint16_t n) {
    for (uint16_t i = 0; i < n; i++) {
        v[i] -= alpha * u[i];
    }
}

/* Public function implementations */

static inline size_t eml_plsr_get_memory_size(
    uint16_t n_samples,
    uint16_t n_features,
    uint16_t n_components
) {
    return EML_PLSR_MEMORY_SIZE(n_samples, n_features, n_components);
}

static inline EmlError eml_plsr_init(
    eml_plsr_t *plsr,
    uint16_t n_samples,
    uint16_t n_features,
    uint16_t n_components,
    void *memory,
    size_t memory_size
) {
    if (!plsr || !memory) {
        return EmlUninitialized;
    }
    
    if (n_samples == 0 || n_features == 0 || n_components == 0 ||
        n_components > n_features || n_components > n_samples) {
        return EmlUnsupported;
    }
    
    if (((uintptr_t)memory & 0x3) != 0) {
        return EmlSizeMismatch;
    }
    
    size_t required = EML_PLSR_MEMORY_SIZE(n_samples, n_features, n_components);
    if (memory_size < required) {
        return EmlSizeMismatch;
    }
    
    plsr->n_samples = n_samples;
    plsr->n_features = n_features;
    plsr->n_components = n_components;
    
    float *mem = (float*)memory;
    size_t offset = 0;
    
    plsr->inputs_work = &mem[offset]; 
    offset += (size_t)n_samples * (size_t)n_features;
    
    plsr->targets_work = &mem[offset]; 
    offset += n_samples;
    
    plsr->weights = &mem[offset]; 
    offset += (size_t)n_features * (size_t)n_components;
    
    plsr->loadings_x = &mem[offset]; 
    offset += (size_t)n_features * (size_t)n_components;
    
    plsr->loadings_y = &mem[offset]; 
    offset += n_components;
    
    plsr->scores = &mem[offset]; 
    offset += (size_t)n_samples * (size_t)n_components;
    
    plsr->score_curr = &mem[offset]; 
    offset += n_samples;
    
    plsr->score_y_curr = &mem[offset]; 
    offset += n_samples;
    
    plsr->weight_curr = &mem[offset]; 
    offset += n_features;
    
    plsr->loading_x_curr = &mem[offset]; 
    offset += n_features;
    
    plsr->current_component = 0;
    plsr->current_iter = 0;
    plsr->component_converged = false;
    plsr->convergence_metric = 0.0f;
    
    return EmlOk;
}

static inline EmlError eml_plsr_fit_start(
    eml_plsr_t *plsr,
    const float *X,
    const float *y
) {
    if (!plsr || !X || !y) {
        return EmlUninitialized;
    }
    
    memcpy(plsr->inputs_work, X, 
           (size_t)plsr->n_samples * (size_t)plsr->n_features * sizeof(float));
    memcpy(plsr->targets_work, y, plsr->n_samples * sizeof(float));
    memcpy(plsr->score_y_curr, plsr->targets_work, plsr->n_samples * sizeof(float));
    
    plsr->current_component = 0;
    plsr->current_iter = 0;
    plsr->component_converged = false;
    
    return EmlOk;
}

static inline EmlError eml_plsr_iteration_step(
    eml_plsr_t *plsr,
    float tolerance
) {
    if (!plsr) {
        return EmlUninitialized;
    }
    
    float score_prev[plsr->n_samples];
    if (plsr->current_iter > 0) {
        memcpy(score_prev, plsr->score_curr, plsr->n_samples * sizeof(float));
    }
    
    float score_y_norm_sq = eml_plsr_dot_product(plsr->score_y_curr, plsr->score_y_curr, plsr->n_samples);
    if (score_y_norm_sq < 1e-10f) {
        return EmlPostconditionFailed;
    }
    
    eml_plsr_mat_trans_vec_mult(plsr->inputs_work, plsr->score_y_curr, plsr->weight_curr, 
                                 plsr->n_samples, plsr->n_features);
    
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->weight_curr[i] /= score_y_norm_sq;
    }
    
    float weight_norm = eml_plsr_normalize_vector(plsr->weight_curr, plsr->n_features);
    if (weight_norm < 1e-10f) {
        return EmlPostconditionFailed;
    }
    
    eml_plsr_mat_vec_mult(plsr->inputs_work, plsr->weight_curr, plsr->score_curr,
                          plsr->n_samples, plsr->n_features);
    
    float score_norm_sq = eml_plsr_dot_product(plsr->score_curr, plsr->score_curr, plsr->n_samples);
    if (score_norm_sq < 1e-10f) {
        return EmlPostconditionFailed;
    }
    
    float loading_y = eml_plsr_dot_product(plsr->targets_work, plsr->score_curr, plsr->n_samples) / score_norm_sq;
    
    for (uint16_t i = 0; i < plsr->n_samples; i++) {
        plsr->score_y_curr[i] = plsr->targets_work[i] * loading_y;
    }
    
    eml_plsr_normalize_vector(plsr->score_y_curr, plsr->n_samples);
    
    if (plsr->current_iter > 0) {
        float diff = 0.0f;
        for (uint16_t i = 0; i < plsr->n_samples; i++) {
            float d = plsr->score_curr[i] - score_prev[i];
            diff += d * d;
        }
        plsr->convergence_metric = sqrtf(diff);
        
        if (plsr->convergence_metric < tolerance) {
            plsr->component_converged = true;
        }
    }
    
    plsr->current_iter++;
    
    return EmlOk;
}

static inline bool eml_plsr_is_converged(const eml_plsr_t *plsr) {
    if (!plsr) {
        return false;
    }
    return plsr->component_converged;
}

static inline EmlError eml_plsr_finalize_component(eml_plsr_t *plsr) {
    if (!plsr) {
        return EmlUninitialized;
    }
    
    if (!plsr->component_converged) {
        return EmlPostconditionFailed;
    }
    
    uint16_t comp = plsr->current_component;
    
    float score_norm_sq = eml_plsr_dot_product(plsr->score_curr, plsr->score_curr, plsr->n_samples);
    if (score_norm_sq < 1e-10f) {
        return EmlPostconditionFailed;
    }
    
    eml_plsr_mat_trans_vec_mult(plsr->inputs_work, plsr->score_curr, plsr->loading_x_curr,
                                 plsr->n_samples, plsr->n_features);
    
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->loading_x_curr[i] /= score_norm_sq;
    }
    
    float loading_y = eml_plsr_dot_product(plsr->targets_work, plsr->score_curr, plsr->n_samples) / score_norm_sq;
    
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->weights[i * plsr->n_components + comp] = plsr->weight_curr[i];
        plsr->loadings_x[i * plsr->n_components + comp] = plsr->loading_x_curr[i];
    }
    plsr->loadings_y[comp] = loading_y;
    
    for (uint16_t i = 0; i < plsr->n_samples; i++) {
        plsr->scores[i * plsr->n_components + comp] = plsr->score_curr[i];
    }
    
    eml_plsr_deflate_matrix(plsr->inputs_work, plsr->score_curr, plsr->loading_x_curr, 1.0f,
                            plsr->n_samples, plsr->n_features);
    
    eml_plsr_deflate_vector(plsr->targets_work, plsr->score_curr, loading_y, plsr->n_samples);
    
    plsr->current_component++;
    plsr->current_iter = 0;
    plsr->component_converged = false;
    
    if (plsr->current_component < plsr->n_components) {
        memcpy(plsr->score_y_curr, plsr->targets_work, plsr->n_samples * sizeof(float));
    }
    
    return EmlOk;
}

static inline bool eml_plsr_is_complete(const eml_plsr_t *plsr) {
    if (!plsr) {
        return false;
    }
    return plsr->current_component >= plsr->n_components;
}

static inline EmlError eml_plsr_predict(
    const eml_plsr_t *plsr,
    const float *x,
    float *y_pred
) {
    if (!plsr || !x || !y_pred) {
        return EmlUninitialized;
    }
    
    *y_pred = 0.0f;
    
    for (uint16_t comp = 0; comp < plsr->current_component; comp++) {
        float score = 0.0f;
        for (uint16_t i = 0; i < plsr->n_features; i++) {
            score += x[i] * plsr->weights[i * plsr->n_components + comp];
        }
        *y_pred += score * plsr->loadings_y[comp];
    }
    
    return EmlOk;
}

static inline EmlError eml_plsr_fit(
    eml_plsr_t *plsr,
    const float *X,
    const float *y,
    uint16_t max_iter,
    float tolerance
) {
    if (!plsr || !X || !y) {
        return EmlUninitialized;
    }
    
    EmlError err = eml_plsr_fit_start(plsr, X, y);
    if (err != EmlOk) {
        return err;
    }
    
    while (!eml_plsr_is_complete(plsr)) {
        while (!eml_plsr_is_converged(plsr) && plsr->current_iter < max_iter) {
            err = eml_plsr_iteration_step(plsr, tolerance);
            if (err != EmlOk) {
                return err;
            }
        }
        
        if (!eml_plsr_is_converged(plsr)) {
            return EmlPostconditionFailed;
        }
        
        err = eml_plsr_finalize_component(plsr);
        if (err != EmlOk) {
            return err;
        }
    }
    
    return EmlOk;
}

#endif /* EML_PLSR_IMPLEMENTATION */

#endif /* EML_PLSR_H */
