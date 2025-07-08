#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define DEBUG 0

#if DEBUG
#define printf(fmt, ...) mp_printf(&mp_plat_print, fmt, ##__VA_ARGS__)
#else
#define printf(fmt, ...) ((void)0)
#endif

typedef struct _EmlTreesNode {
    int8_t feature;   // -1 for leaf nodes
    int16_t value;    // threshold or class label
    int16_t left;     // left child index
    int16_t right;    // right child index
} EmlTreesNode;

typedef struct _NodeState {
    int16_t node_idx;     // current node being processed
    int16_t start;        // sample range start
    int16_t end;          // sample range end
    int16_t depth;        // current depth
} NodeState;

typedef struct _EmlTreesConfig {
    int16_t max_depth;
    int16_t min_samples_leaf;
    int16_t n_thresholds;
    float subsample_ratio;          // subsample ratio as float (0.0 to 1.0)
    float feature_subsample_ratio;  // feature subsample ratio as float (0.0 to 1.0)
    uint32_t rng_seed;
} EmlTreesConfig;

typedef struct _EmlTreesModel {
    EmlTreesNode *nodes;          // Pre-allocated node array
    int16_t *tree_starts;         // Start index for each tree
    int16_t max_nodes;            // Maximum nodes available
    int16_t n_nodes_used;         // Current nodes used
    int16_t n_features;           // Number of features
    int16_t n_classes;            // Number of classes
    int16_t n_trees;              // Number of trees
    EmlTreesConfig config;
} EmlTreesModel;

typedef struct _EmlTreesWorkspace {
    int16_t *sample_indices;      // Sample indices for current tree
    int16_t *feature_indices;     // Feature indices for current tree
    int16_t *min_vals;            // Min values per feature [n_features]
    int16_t *max_vals;            // Max values per feature [n_features]
    NodeState *node_stack;        // Stack for tree building
    uint32_t rng_state;           // Simple RNG state
    int16_t n_samples;            // Number of samples
} EmlTreesWorkspace;

// Simple linear congruential generator
static uint32_t eml_rand(uint32_t *state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

// Fisher-Yates shuffle for subsampling
static void shuffle_indices(int16_t *indices, int16_t n, uint32_t *rng_state) {
    for (int16_t i = 0; i < n - 1; i++) {
        int16_t j = i + (eml_rand(rng_state) % (n - i));
        int16_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Calculate Gini impurity from class counts
static float calculate_gini_from_counts(const int16_t *counts, int16_t total, int16_t n_classes) {
    if (total == 0) return 0.0f;
    
    float gini = 1.0f;
    for (int16_t i = 0; i < n_classes; i++) {
        if (counts[i] > 0) {
            float prob = (float)counts[i] / (float)total;
            gini -= prob * prob;
        }
    }
    
    return gini;
}




// Partition samples based on feature threshold
static int16_t partition_samples(const int16_t *features, EmlTreesModel *model, 
                                EmlTreesWorkspace *workspace, int16_t start, int16_t end, 
                                int8_t feature, int16_t threshold) {
    int16_t left = start;
    int16_t right = end - 1;
    
    while (left <= right) {
        // Find element on left that should be on right
        while (left <= right && features[workspace->sample_indices[left] * model->n_features + feature] <= threshold) {
            left++;
        }
        
        // Find element on right that should be on left
        while (left <= right && features[workspace->sample_indices[right] * model->n_features + feature] > threshold) {
            right--;
        }
        
        // Swap if needed
        if (left < right) {
            int16_t temp = workspace->sample_indices[left];
            workspace->sample_indices[left] = workspace->sample_indices[right];
            workspace->sample_indices[right] = temp;
            left++;
            right--;
        }
    }
    
    return left; // Split point
}





// Add this debug version of eml_trees_predict_proba
int16_t eml_trees_predict_proba(const EmlTreesModel *model, const int16_t *features, 
                               float *probabilities, int16_t *votes) {
    
    // Initialize vote counts
    for (int16_t i = 0; i < model->n_classes; i++) {
        votes[i] = 0;
    }
    
    //printf("Prediction debug: features=[%d,%d]\n", features[0], features[1]);
    
    // Get prediction from each tree
    for (int16_t tree = 0; tree < model->n_trees; tree++) {
        int16_t node_idx = model->tree_starts[tree];
        //printf("  Tree %d: starting at node %d\n", tree, node_idx);
        
        // Traverse tree
        int16_t steps = 0;
        while (node_idx >= 0 && node_idx < model->n_nodes_used && 
               model->nodes[node_idx].feature != -1 && steps < 20) {
            
            int8_t feature = model->nodes[node_idx].feature;
            int16_t threshold = model->nodes[node_idx].value;
            int16_t left = model->nodes[node_idx].left;
            int16_t right = model->nodes[node_idx].right;
            
            printf("    Node %d: feature=%d, threshold=%d, feature_val=%d\n", 
                   node_idx, feature, threshold, features[feature]);
            
            if (features[feature] <= threshold) {
                //printf("    Going LEFT to node %d\n", left);
                node_idx = left;
            } else {
                //printf("    Going RIGHT to node %d\n", right);
                node_idx = right;
            }
            steps++;
        }
        
        // Check leaf node
        if (node_idx >= 0 && node_idx < model->n_nodes_used) {
            int16_t predicted_class = model->nodes[node_idx].value;
            //printf("  Tree %d: reached leaf node %d, class=%d\n", tree, node_idx, predicted_class);
            
            if (predicted_class >= 0 && predicted_class < model->n_classes) {
                votes[predicted_class]++;
            }
        } else {
            //printf("  Tree %d: invalid leaf node %d\n", tree, node_idx);
        }
    }
    
    //printf("Final votes: [%d,%d]\n", votes[0], votes[1]);
    
    // Rest of function unchanged...
    for (int16_t i = 0; i < model->n_classes; i++) {
        probabilities[i] = (float)votes[i] / (float)model->n_trees;
    }
    
    int16_t max_votes = 0;
    int16_t predicted_class = 0;
    for (int16_t i = 0; i < model->n_classes; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}




// ALSO: Make sure get_majority_class is working correctly
static int16_t get_majority_class(const int16_t *labels, const int16_t *indices,
                                 int16_t start, int16_t end, int16_t n_classes) {
    int16_t counts[256] = {0};
    int16_t max_count = 0;
    int16_t majority_class = 0;
    
    printf("get_majority_class: samples %d to %d\n", start, end-1);
    
    if (start >= end) {
        printf("  No samples, returning class 0\n");
        return 0;
    }
    
    // Count occurrences
    for (int16_t i = start; i < end; i++) {
        int16_t sample_idx = indices[i];
        int16_t label = labels[sample_idx];
        
        if (label >= 0 && label < n_classes) {
            counts[label]++;
            printf("  Sample %d: index=%d, label=%d\n", i, sample_idx, label);
        }
    }
    
    // Find majority
    for (int16_t i = 0; i < n_classes; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            majority_class = i;
        }
    }
    
    printf("  Counts: [%d,%d], majority class: %d\n", counts[0], counts[1], majority_class);
    
    return majority_class;
}


// CRITICAL FIX: Accept splits with zero improvement
// For complex patterns like XOR, we need to allow splits that don't immediately improve Gini
// but will lead to better splits at deeper levels

static int16_t find_best_split(const int16_t *features, const int16_t *labels,
                              EmlTreesModel *model, EmlTreesWorkspace *workspace, 
                              int16_t start, int16_t end, int16_t n_features_subset, 
                              int8_t *best_feature, int16_t *best_threshold) {
    
    float best_improvement = -1.0f;
    *best_feature = -1;
    *best_threshold = 0;
    
    int16_t total_samples = end - start;
    
    if (total_samples < 2) {
        return -1;
    }
    
    // Calculate parent class distribution
    int16_t parent_counts[256] = {0};
    for (int16_t i = start; i < end; i++) {
        int16_t sample_idx = workspace->sample_indices[i];
        int16_t label = labels[sample_idx];
        parent_counts[label]++;
    }
    
    float parent_gini = calculate_gini_from_counts(parent_counts, total_samples, model->n_classes);
    
    // If already pure, no split needed
    if (parent_gini == 0.0f) {
        return -1;
    }
    
    // Try each feature
    for (int16_t f = 0; f < n_features_subset; f++) {
        int16_t feature_idx = workspace->feature_indices[f];
        
        // Collect unique values for this feature in this node
        int16_t unique_vals[50];
        int16_t n_unique = 0;
        
        for (int16_t i = start; i < end; i++) {
            int16_t sample_idx = workspace->sample_indices[i];
            int16_t val = features[sample_idx * model->n_features + feature_idx];
            
            // Check if already in unique_vals
            bool already_present = false;
            for (int16_t u = 0; u < n_unique; u++) {
                if (unique_vals[u] == val) {
                    already_present = true;
                    break;
                }
            }
            if (!already_present && n_unique < 50) {
                unique_vals[n_unique++] = val;
            }
        }
        
        if (n_unique < 2) {
            continue; // Need at least 2 unique values to split
        }
        
        // Sort unique values to try thresholds between them
        for (int16_t i = 0; i < n_unique - 1; i++) {
            for (int16_t j = i + 1; j < n_unique; j++) {
                if (unique_vals[i] > unique_vals[j]) {
                    int16_t temp = unique_vals[i];
                    unique_vals[i] = unique_vals[j];
                    unique_vals[j] = temp;
                }
            }
        }
        
        // Try thresholds between consecutive unique values
        for (int16_t u = 0; u < n_unique - 1; u++) {
            // Use threshold between unique_vals[u] and unique_vals[u+1]
            int16_t threshold = unique_vals[u];
            
            // Count left/right distributions
            int16_t left_counts[256] = {0};
            int16_t right_counts[256] = {0};
            int16_t left_total = 0, right_total = 0;
            
            for (int16_t i = start; i < end; i++) {
                int16_t sample_idx = workspace->sample_indices[i];
                int16_t feature_val = features[sample_idx * model->n_features + feature_idx];
                int16_t label = labels[sample_idx];
                
                if (feature_val <= threshold) {
                    left_counts[label]++;
                    left_total++;
                } else {
                    right_counts[label]++;
                    right_total++;
                }
            }
            
            // Check if split creates non-empty partitions
            if (left_total == 0 || right_total == 0) {
                continue;
            }
            
            // Check min_samples_leaf constraint
            if (left_total < model->config.min_samples_leaf || right_total < model->config.min_samples_leaf) {
                continue;
            }
            
            // Calculate improvement
            float left_gini = calculate_gini_from_counts(left_counts, left_total, model->n_classes);
            float right_gini = calculate_gini_from_counts(right_counts, right_total, model->n_classes);
            float weighted_gini = ((float)left_total * left_gini + (float)right_total * right_gini) / (float)total_samples;
            float improvement = parent_gini - weighted_gini;
            
            // CRITICAL FIX: Accept splits with improvement >= 0.0 (not just > 0.0)
            // This allows splits that don't immediately improve but may lead to better deeper splits
            if (improvement >= best_improvement) {
                best_improvement = improvement;
                *best_feature = feature_idx;
                *best_threshold = threshold;
            }
        }
    }
    
    // CRITICAL FIX: Accept any valid split, even with zero improvement
    // Change the return condition to accept improvement >= 0.0
    return (*best_feature != -1) ? 0 : -1;
}

// ALSO: Ensure stopping criteria allow deep enough trees for XOR
static int16_t build_tree(EmlTreesModel *model, EmlTreesWorkspace *workspace,
                         const int16_t *features, const int16_t *labels) {
    
    int16_t tree_start = model->n_nodes_used;
    
    // Subsample features
    int16_t n_features_subset = (int16_t)((float)model->n_features * model->config.feature_subsample_ratio);
    if (n_features_subset < 1) n_features_subset = 1;
    if (n_features_subset > model->n_features) n_features_subset = model->n_features;
    
    for (int16_t i = 0; i < model->n_features; i++) {
        workspace->feature_indices[i] = i;
    }
    shuffle_indices(workspace->feature_indices, model->n_features, &workspace->rng_state);
    
    // Initialize root node state
    int16_t stack_size = 1;
    workspace->node_stack[0].node_idx = tree_start;
    workspace->node_stack[0].start = 0;
    workspace->node_stack[0].end = workspace->n_samples;
    workspace->node_stack[0].depth = 0;
    
    // Process stack
    while (stack_size > 0) {
        NodeState current = workspace->node_stack[--stack_size];
        int16_t node_idx = current.node_idx;
        
        if (node_idx >= model->max_nodes) {
            return -1;
        }
        
        // Check stopping criteria - MODIFIED for XOR
        int16_t n_samples_node = current.end - current.start;
        
        // Create leaf if:
        // 1. Reached max depth, OR
        // 2. Too few samples for further splitting, OR  
        // 3. Node is already pure
        bool should_stop = false;
        
        if (current.depth >= model->config.max_depth) {
            should_stop = true;
        } else if (n_samples_node < 2 * model->config.min_samples_leaf) {
            should_stop = true;
        } else {
            // Check if node is pure
            int16_t first_label = -1;
            bool is_pure = true;
            for (int16_t i = current.start; i < current.end; i++) {
                int16_t sample_idx = workspace->sample_indices[i];
                int16_t label = labels[sample_idx];
                if (first_label == -1) {
                    first_label = label;
                } else if (label != first_label) {
                    is_pure = false;
                    break;
                }
            }
            if (is_pure) {
                should_stop = true;
            }
        }
        
        if (should_stop) {
            // Create leaf node
            int16_t majority = get_majority_class(labels, workspace->sample_indices,
                                                 current.start, current.end, model->n_classes);
            
            model->nodes[node_idx].feature = -1;
            model->nodes[node_idx].value = majority;
            model->nodes[node_idx].left = -1;
            model->nodes[node_idx].right = -1;
            
            if (node_idx >= model->n_nodes_used) {
                model->n_nodes_used = node_idx + 1;
            }
            continue;
        }
        
        // Find best split
        int8_t best_feature;
        int16_t best_threshold;
        int16_t split_result = find_best_split(features, labels, model, workspace, 
                                              current.start, current.end,
                                              n_features_subset, &best_feature, &best_threshold);
        
        if (split_result != 0 || best_feature == -1) {
            // No valid split found, create leaf
            int16_t majority = get_majority_class(labels, workspace->sample_indices,
                                                 current.start, current.end, model->n_classes);
            
            model->nodes[node_idx].feature = -1;
            model->nodes[node_idx].value = majority;
            model->nodes[node_idx].left = -1;
            model->nodes[node_idx].right = -1;
            
            if (node_idx >= model->n_nodes_used) {
                model->n_nodes_used = node_idx + 1;
            }
            continue;
        }
        
        // Partition samples
        int16_t split_point = partition_samples(features, model, workspace, current.start, current.end, 
                                               best_feature, best_threshold);
        
        if (split_point <= current.start || split_point >= current.end) {
            // Partition failed, create leaf
            int16_t majority = get_majority_class(labels, workspace->sample_indices,
                                                 current.start, current.end, model->n_classes);
            
            model->nodes[node_idx].feature = -1;
            model->nodes[node_idx].value = majority;
            model->nodes[node_idx].left = -1;
            model->nodes[node_idx].right = -1;
            
            if (node_idx >= model->n_nodes_used) {
                model->n_nodes_used = node_idx + 1;
            }
            continue;
        }
        
        // Calculate next available node indices
        int16_t next_node = model->n_nodes_used;
        if (next_node <= node_idx) {
            next_node = node_idx + 1;
        }
        
        if (next_node + 1 >= model->max_nodes) {
            // Not enough space, create leaf
            int16_t majority = get_majority_class(labels, workspace->sample_indices,
                                                 current.start, current.end, model->n_classes);
            
            model->nodes[node_idx].feature = -1;
            model->nodes[node_idx].value = majority;
            model->nodes[node_idx].left = -1;
            model->nodes[node_idx].right = -1;
            
            if (node_idx >= model->n_nodes_used) {
                model->n_nodes_used = node_idx + 1;
            }
            continue;
        }
        
        // Create internal node
        model->nodes[node_idx].feature = best_feature;
        model->nodes[node_idx].value = best_threshold;
        model->nodes[node_idx].left = next_node;
        model->nodes[node_idx].right = next_node + 1;
        
        // Update n_nodes_used
        model->n_nodes_used = next_node + 2;
        
        // Add children to stack
        if (stack_size < 98) {
            // Right child
            workspace->node_stack[stack_size].node_idx = model->nodes[node_idx].right;
            workspace->node_stack[stack_size].start = split_point;
            workspace->node_stack[stack_size].end = current.end;
            workspace->node_stack[stack_size].depth = current.depth + 1;
            stack_size++;
            
            // Left child
            workspace->node_stack[stack_size].node_idx = model->nodes[node_idx].left;
            workspace->node_stack[stack_size].start = current.start;
            workspace->node_stack[stack_size].end = split_point;
            workspace->node_stack[stack_size].depth = current.depth + 1;
            stack_size++;
        }
    }
    
    return 0;
}


// CRITICAL: Also check that we're not accidentally filtering out all samples during subsampling
// In eml_trees_train, make sure to print subsample_size:

int16_t eml_trees_train(EmlTreesModel *model, EmlTreesWorkspace *workspace,
                       const int16_t *features, const int16_t *labels) {
    
    model->n_nodes_used = 0;
    workspace->rng_state = model->config.rng_seed;
    
    printf("Training: %d trees, %d total samples\n", model->n_trees, workspace->n_samples);
    
    // Calculate subsample size
    int16_t subsample_size = (int16_t)((float)workspace->n_samples * model->config.subsample_ratio);
    if (subsample_size < 1) subsample_size = 1;
    if (subsample_size > workspace->n_samples) subsample_size = workspace->n_samples;
    
    printf("Subsample size: %d (ratio=%.2f)\n", subsample_size, model->config.subsample_ratio);
    
    // Initialize sample indices
    for (int16_t i = 0; i < workspace->n_samples; i++) {
        workspace->sample_indices[i] = i;
    }
    
    // Build each tree
    for (int16_t tree = 0; tree < model->n_trees; tree++) {
        printf("\n=== Building tree %d ===\n", tree);
        
        model->tree_starts[tree] = model->n_nodes_used;
        
        // Subsample
        shuffle_indices(workspace->sample_indices, workspace->n_samples, &workspace->rng_state);
        
        printf("After shuffle, first few indices: ");
        for (int16_t i = 0; i < (workspace->n_samples < 8 ? workspace->n_samples : 8); i++) {
            printf("%d ", workspace->sample_indices[i]);
        }
        printf("\n");
        
        // Set subsample size
        int16_t original_n_samples = workspace->n_samples;
        workspace->n_samples = subsample_size;
        
        printf("Using %d samples for this tree\n", workspace->n_samples);
        
        // Build tree
        int16_t result = build_tree(model, workspace, features, labels);
        
        // Restore original sample count
        workspace->n_samples = original_n_samples;
        
        if (result != 0) {
            return result;
        }
    }
    
    printf("\nTraining completed: %d nodes total\n", model->n_nodes_used);
    return 0;
}

