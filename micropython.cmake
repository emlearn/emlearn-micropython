add_library(usermod_emlearn INTERFACE)

target_sources(usermod_emlearn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/trees.c
)

target_include_directories(usermod_emlearn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE usermod_emlearn)
