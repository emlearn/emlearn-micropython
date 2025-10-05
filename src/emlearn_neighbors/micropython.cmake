add_library(usermod_emlearn_neighbors INTERFACE)

execute_process(
    COMMAND python3 -c "import emlearn; print(emlearn.includedir)"
    OUTPUT_VARIABLE EMLEARN_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_sources(usermod_emlearn_neighbors INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/neighbors.c
)

target_include_directories(usermod_emlearn_neighbors INTERFACE
    ${EMLEARN_DIR}
)

target_link_libraries(usermod INTERFACE usermod_emlearn_neighbors)
