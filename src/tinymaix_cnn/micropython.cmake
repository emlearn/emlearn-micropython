add_library(usermod_cnn INTERFACE)

execute_process(
    COMMAND python3 -c "import emlearn; print(emlearn.includedir)"
    OUTPUT_VARIABLE EMLEARN_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_sources(usermod_cnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/mod_cnn.c
)

target_include_directories(usermod_cnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/int8
    ${CMAKE_CURRENT_LIST_DIR}/../../dependencies/TinyMaix/include
    ${CMAKE_CURRENT_LIST_DIR}/../../dependencies/TinyMaix/src
)

target_compile_options(usermod_cnn INTERFACE
    -Wno-error=unused-variable -Wno-error=multichar -Wdouble-promotion
)
target_link_libraries(usermod INTERFACE usermod_cnn)
