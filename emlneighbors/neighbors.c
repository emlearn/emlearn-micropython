
#define EML_LOG_ENABLE 1
#define EML_NEIGHBORS_LOG_LEVEL 0
#include <eml_neighbors.h>


#define N_FEATURES 3
#define MAX_ITEMS 100
#define DATA_LENGTH (MAX_ITEMS*N_FEATURES)

// FIXME: move into emlearn as a test
int
main(int argc, char *argv[])
{
    // Allocate space
    int16_t data[DATA_LENGTH];
    int16_t labels[MAX_ITEMS];
    EmlNeighborsDistanceItem distances[MAX_ITEMS];

    const int K_NEIGHBORS = 10;

    // Setup model
    EmlNeighborsModel _model = { N_FEATURES, 0, MAX_ITEMS, data, labels, K_NEIGHBORS };
    EmlNeighborsModel *model = &_model;

    EmlError check_err = eml_neighbors_check(model, DATA_LENGTH, MAX_ITEMS, MAX_ITEMS);
    if (check_err != EmlOk) {
        return -1;
    }

    // Add data
    for (int i=0; i<MAX_ITEMS-2; i++) {
        const int16_t features[N_FEATURES] = { -21, -30, -20 };
        const int16_t label = 1;
        EmlError add_err = eml_neighbors_add_item(model, features, N_FEATURES, label);
        if (add_err != EmlOk) {
            return -2;
        }
    }
    for (int i=0; i<2; i++) {
        const int16_t features[N_FEATURES] = { 12, 10, 10 };
        const int16_t label = 3;
        EmlError add_err = eml_neighbors_add_item(model, features, N_FEATURES, label);
        if (add_err != EmlOk) {
            return -2;
        }
    }

    // Use the model

    const int16_t features[N_FEATURES] = { 12, 9, 15 };
    //const int16_t features[N_FEATURES] = { -12, -9, -15 };

    int16_t label = -1;
    EmlError predict_err = \
        eml_neighbors_predict(model, features, N_FEATURES, distances, MAX_ITEMS, &label);
    if (predict_err != EmlOk) {
        return -3;
    }

    printf("prediction label=%d \n", label);

    return 0;
}
