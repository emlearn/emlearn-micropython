add_library(usermod_emlearn_kmeans INTERFACE)

target_sources(usermod_emlearn_kmeans INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/kmeans.c
)

target_include_directories(usermod_emlearn_kmeans INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE usermod_emlearn_kmeans)
