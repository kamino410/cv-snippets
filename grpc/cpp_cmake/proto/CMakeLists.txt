find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# v3.14.0ではgRPC::grpc_cpp_pluginのプロパティから実行ファイルのパスを取得できなかった
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

get_filename_component(myproto "./helloworld.proto" ABSOLUTE)
get_filename_component(myproto_path "${myproto}" PATH)

set(myproto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(myproto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
add_custom_command(
      OUTPUT "${myproto_srcs}" "${myproto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${myproto_path}"
        --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
        "${myproto}"
      DEPENDS "${myproto}")

add_library(proto ${hw_grpc_srcs} ${myproto_srcs})
target_include_directories(proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
# /usr/local/lib/cmake/grpc/gRPCTargets.cmake を見ながら他にも追加が必要かも
target_link_libraries(proto PUBLIC gRPC::grpc++ gRPC::grpc++_reflection)

