find_package(Protobuf REQUIRED)

# if(NOT PROTOBUF_PROTOC_EXECUTABLE)
#     message(FATAL_ERROR "Could not find protoc executable (PROTOBUF_PROTOC_EXECUTABLE)")
# endif()
# message(STATUS "Found protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER user.proto)
get_filename_component(PROTO_INCLUDE_DIR ${PROTO_HEADER} DIRECTORY)

message(STATUS ${PROTO_HEADER})
message(STATUS ${PROTO_INCLUDE_DIR})

add_library(proto ${PROTO_SRC})
target_include_directories(proto PUBLIC
    ${PROTOBUF_INCLUDE_DIR}
    ${PROTO_INCLUDE_DIR}
)
target_link_libraries(proto PUBLIC ${PROTOBUF_LIBRARY})

