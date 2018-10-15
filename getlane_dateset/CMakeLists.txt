# Blog: http://blog.csdn.net/fengbingchun/article/details/77193530
# CMake file for CUDA_Test

# 设定依赖的CMake版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
# 指定项目名称
PROJECT(draw)

# 搜索指定的OpenCV外部依赖库头文件和库文件位置
FIND_PACKAGE(OpenCV)

# 打印相关信息, CMAKE_CURRENT_SOURCE_DIR指的是当前处理的CMakeLists.txt所在的路径
MESSAGE(STATUS "current path: ${CMAKE_CURRENT_SOURCE_DIR}")

# 定义用户自定义变量
SET(PATH_CPP_FILES ./codes)
MESSAGE(STATUS "cpp files path: ${PATH_CPP_FILES}")

# 递归查询所有匹配的文件:*.cpp
FILE(GLOB_RECURSE CPP_LIST ${PATH_CPP_FILES}/*.cpp)
MESSAGE(STATUS "cpp list: ${CPP_LIST}")

# 使CMake支持C++11特性
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

# 指定OpenCV头文件搜索路径
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
#MESSAGE(STATUS "opencv include directory: ${OpenCV_INCLUDE_DIRS}")

add_compile_options(-fpermissive -std=c++11) #to avoid the error: taking address of temporary [-fpermissive]

# 编译可执行程序，即生成可执行文件CUDA_Test
ADD_EXECUTABLE(draw ${CPP_LIST})
# 用来为target添加需要链接的共享库，指定工程所用的依赖库，包括动态库和静态库
TARGET_LINK_LIBRARIES(draw ${OpenCV_LIBS})
#MESSAGE(STATUS "opencv libs: ${OpenCV_LIBS}")

