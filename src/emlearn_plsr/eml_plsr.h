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
 * For PLS1 (single response variable), the NIPALS algorithm converges in
 * exactly one iteration per component. Data is automatically centered
 * before fitting, matching sklearn's PLSRegression behavior.
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
      (size_t)(n_features) +                        /* weight_curr */ \
      (size_t)(n_features) +                        /* loading_x_curr */ \
      (size_t)(n_features) +                        /* x_mean */ \
      (size_t)(n_features)                          /* x_work (predict) */ \
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
    float *weight_curr;
    float *loading_x_curr;
    float *x_mean;
    float *x_work;       /* temp buffer for prediction */

    /* Centering offsets (stored for prediction) */
    float y_mean;
    
    /* Auto-centering control (default: enabled) */
    bool auto_center;
    
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

/* Debug output control */
#ifndef EMLEARN_PLSR_DEBUG
#define EMLEARN_PLSR_DEBUG 0
#endif

#if EMLEARN_PLSR_DEBUG
#include <stdio.h>
#define EMLEARN_PLSR_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define EMLEARN_PLSR_PRINTF(fmt, ...)
#endif

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
    
    plsr->weight_curr = &mem[offset]; 
    offset += n_features;
    
    plsr->loading_x_curr = &mem[offset]; 
    offset += n_features;
    
    plsr->x_mean = &mem[offset]; 
    offset += n_features;
    
    plsr->x_work = &mem[offset]; 
    offset += n_features;
    
    plsr->y_mean = 0.0f;
    plsr->auto_center = true;
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
    
    const uint16_t n = plsr->n_samples;
    const uint16_t m = plsr->n_features;
    
    /* Copy X into work buffer */
    memcpy(plsr->inputs_work, X, (size_t)n * (size_t)m * sizeof(float));
    
    /* Copy y into work buffer */
    memcpy(plsr->targets_work, y, n * sizeof(float));
    
    if (plsr->auto_center) {
        /* Compute and store X column means */
        for (uint16_t j = 0; j < m; j++) {
            float sum = 0.0f;
            for (uint16_t i = 0; i < n; i++) {
                sum += plsr->inputs_work[(size_t)i * m + j];
            }
            plsr->x_mean[j] = sum / (float)n;
        }
        
        /* Center X */
        for (uint16_t i = 0; i < n; i++) {
            for (uint16_t j = 0; j < m; j++) {
                plsr->inputs_work[(size_t)i * m + j] -= plsr->x_mean[j];
            }
        }
        
        /* Compute and store y mean */
        float y_sum = 0.0f;
        for (uint16_t i = 0; i < n; i++) {
            y_sum += plsr->targets_work[i];
        }
        plsr->y_mean = y_sum / (float)n;
        
        /* Center y */
        for (uint16_t i = 0; i < n; i++) {
            plsr->targets_work[i] -= plsr->y_mean;
        }
    } else {
        /* No centering: zero out means */
        memset(plsr->x_mean, 0, m * sizeof(float));
        plsr->y_mean = 0.0f;
    }
    
    plsr->current_component = 0;
    plsr->current_iter = 0;
    plsr->component_converged = false;
    plsr->convergence_metric = 0.0f;
    
    EMLEARN_PLSR_PRINTF("plsr fit_start: n=%d m=%d y_mean=%.6f x_mean[0]=%.6f\n",
        n, m, plsr->y_mean, plsr->x_mean[0]);
    
    return EmlOk;
}

/**
 * @brief Perform one NIPALS iteration step for PLS1
 * 
 * For PLS1 (single y), the NIPALS algorithm converges in exactly one
 * iteration per component. The weight vector w is computed directly
 * from the covariance X^T y, then normalized.
 */
static inline EmlError eml_plsr_iteration_step(
    eml_plsr_t *plsr,
    float tolerance
) {
    if (!plsr) {
        return EmlUninitialized;
    }
    
    /* w = X^T y_residual (covariance direction) */
    eml_plsr_mat_trans_vec_mult(plsr->inputs_work, plsr->targets_work, plsr->weight_curr,
                                 plsr->n_samples, plsr->n_features);
    
    /* Normalize w */
    float weight_norm = eml_plsr_normalize_vector(plsr->weight_curr, plsr->n_features);
    if (weight_norm < 1e-10f) {
        return EmlPostconditionFailed;
    }
    
    /* t = X w (scores) */
    eml_plsr_mat_vec_mult(plsr->inputs_work, plsr->weight_curr, plsr->score_curr,
                          plsr->n_samples, plsr->n_features);
    
    EMLEARN_PLSR_PRINTF("plsr step: comp=%d w_norm=%.6f t_norm=%.4f\n",
        plsr->current_component, weight_norm,
        eml_plsr_vector_norm(plsr->score_curr, plsr->n_samples));
    
    /* For PLS1, convergence is immediate */
    plsr->convergence_metric = 0.0f;
    plsr->component_converged = true;
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
    
    /* p = X^T t / (t^T t) -- X loadings */
    eml_plsr_mat_trans_vec_mult(plsr->inputs_work, plsr->score_curr, plsr->loading_x_curr,
                                 plsr->n_samples, plsr->n_features);
    
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->loading_x_curr[i] /= score_norm_sq;
    }
    
    /* c = y^T t / (t^T t) -- y loading */
    float loading_y = eml_plsr_dot_product(plsr->targets_work, plsr->score_curr, plsr->n_samples) / score_norm_sq;
    
    /* Store component results */
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->weights[i * plsr->n_components + comp] = plsr->weight_curr[i];
        plsr->loadings_x[i * plsr->n_components + comp] = plsr->loading_x_curr[i];
    }
    plsr->loadings_y[comp] = loading_y;
    
    for (uint16_t i = 0; i < plsr->n_samples; i++) {
        plsr->scores[i * plsr->n_components + comp] = plsr->score_curr[i];
    }
    
    EMLEARN_PLSR_PRINTF("plsr finalize: comp=%d c=%.6f t_norm=%.4f\n",
        comp, loading_y, sqrtf(score_norm_sq));
    
    /* Deflate X: X = X - t p^T */
    eml_plsr_deflate_matrix(plsr->inputs_work, plsr->score_curr, plsr->loading_x_curr, 1.0f,
                            plsr->n_samples, plsr->n_features);
    
    /* Deflate y: y = y - c t */
    eml_plsr_deflate_vector(plsr->targets_work, plsr->score_curr, loading_y, plsr->n_samples);
    
    plsr->current_component++;
    plsr->current_iter = 0;
    plsr->component_converged = false;
    
    return EmlOk;
}

static inline bool eml_plsr_is_complete(const eml_plsr_t *plsr) {
    if (!plsr) {
        return false;
    }
    return plsr->current_component >= plsr->n_components;
}

/**
 * @brief Predict using the PLSR model with incremental deflation
 * 
 * For each component k:
 *   score_k = x_centered . w_k
 *   y_pred += score_k * c_k
 *   x_centered -= score_k * p_k  (deflate)
 * Finally: y_pred += y_mean
 */
static inline EmlError eml_plsr_predict(
    const eml_plsr_t *plsr,
    const float *x,
    float *y_pred
) {
    if (!plsr || !x || !y_pred) {
        return EmlUninitialized;
    }
    
    /* Center input: x_work = x - x_mean */
    for (uint16_t i = 0; i < plsr->n_features; i++) {
        plsr->x_work[i] = x[i] - plsr->x_mean[i];
    }
    
    /* Incremental deflation prediction */
    *y_pred = 0.0f;
    
    for (uint16_t comp = 0; comp < plsr->current_component; comp++) {
        /* score = x_work . w_comp */
        float score = 0.0f;
        for (uint16_t i = 0; i < plsr->n_features; i++) {
            score += plsr->x_work[i] * plsr->weights[i * plsr->n_components + comp];
        }
        
        /* y_pred += score * c_comp */
        *y_pred += score * plsr->loadings_y[comp];
        
        /* x_work -= score * p_comp (deflate for next component) */
        for (uint16_t i = 0; i < plsr->n_features; i++) {
            plsr->x_work[i] -= score * plsr->loadings_x[i * plsr->n_components + comp];
        }
    }
    
    /* Add back y mean */
    *y_pred += plsr->y_mean;
    
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
