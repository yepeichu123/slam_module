# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ypc/xiaoc/code/slam_module/ICP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ypc/xiaoc/code/slam_module/ICP/build

# Include any dependencies generated for this target.
include CMakeFiles/ICP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ICP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ICP.dir/flags.make

CMakeFiles/ICP.dir/src/main.cpp.o: CMakeFiles/ICP.dir/flags.make
CMakeFiles/ICP.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypc/xiaoc/code/slam_module/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ICP.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ICP.dir/src/main.cpp.o -c /home/ypc/xiaoc/code/slam_module/ICP/src/main.cpp

CMakeFiles/ICP.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ICP.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypc/xiaoc/code/slam_module/ICP/src/main.cpp > CMakeFiles/ICP.dir/src/main.cpp.i

CMakeFiles/ICP.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ICP.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypc/xiaoc/code/slam_module/ICP/src/main.cpp -o CMakeFiles/ICP.dir/src/main.cpp.s

CMakeFiles/ICP.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/ICP.dir/src/main.cpp.o.requires

CMakeFiles/ICP.dir/src/main.cpp.o.provides: CMakeFiles/ICP.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ICP.dir/build.make CMakeFiles/ICP.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/ICP.dir/src/main.cpp.o.provides

CMakeFiles/ICP.dir/src/main.cpp.o.provides.build: CMakeFiles/ICP.dir/src/main.cpp.o


CMakeFiles/ICP.dir/src/ICP.cpp.o: CMakeFiles/ICP.dir/flags.make
CMakeFiles/ICP.dir/src/ICP.cpp.o: ../src/ICP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypc/xiaoc/code/slam_module/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ICP.dir/src/ICP.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ICP.dir/src/ICP.cpp.o -c /home/ypc/xiaoc/code/slam_module/ICP/src/ICP.cpp

CMakeFiles/ICP.dir/src/ICP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ICP.dir/src/ICP.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypc/xiaoc/code/slam_module/ICP/src/ICP.cpp > CMakeFiles/ICP.dir/src/ICP.cpp.i

CMakeFiles/ICP.dir/src/ICP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ICP.dir/src/ICP.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypc/xiaoc/code/slam_module/ICP/src/ICP.cpp -o CMakeFiles/ICP.dir/src/ICP.cpp.s

CMakeFiles/ICP.dir/src/ICP.cpp.o.requires:

.PHONY : CMakeFiles/ICP.dir/src/ICP.cpp.o.requires

CMakeFiles/ICP.dir/src/ICP.cpp.o.provides: CMakeFiles/ICP.dir/src/ICP.cpp.o.requires
	$(MAKE) -f CMakeFiles/ICP.dir/build.make CMakeFiles/ICP.dir/src/ICP.cpp.o.provides.build
.PHONY : CMakeFiles/ICP.dir/src/ICP.cpp.o.provides

CMakeFiles/ICP.dir/src/ICP.cpp.o.provides.build: CMakeFiles/ICP.dir/src/ICP.cpp.o


CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o: CMakeFiles/ICP.dir/flags.make
CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o: ../src/ICP_G2O.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypc/xiaoc/code/slam_module/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o -c /home/ypc/xiaoc/code/slam_module/ICP/src/ICP_G2O.cpp

CMakeFiles/ICP.dir/src/ICP_G2O.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ICP.dir/src/ICP_G2O.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypc/xiaoc/code/slam_module/ICP/src/ICP_G2O.cpp > CMakeFiles/ICP.dir/src/ICP_G2O.cpp.i

CMakeFiles/ICP.dir/src/ICP_G2O.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ICP.dir/src/ICP_G2O.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypc/xiaoc/code/slam_module/ICP/src/ICP_G2O.cpp -o CMakeFiles/ICP.dir/src/ICP_G2O.cpp.s

CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.requires:

.PHONY : CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.requires

CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.provides: CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.requires
	$(MAKE) -f CMakeFiles/ICP.dir/build.make CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.provides.build
.PHONY : CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.provides

CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.provides.build: CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o


# Object files for target ICP
ICP_OBJECTS = \
"CMakeFiles/ICP.dir/src/main.cpp.o" \
"CMakeFiles/ICP.dir/src/ICP.cpp.o" \
"CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o"

# External object files for target ICP
ICP_EXTERNAL_OBJECTS =

../bin/ICP: CMakeFiles/ICP.dir/src/main.cpp.o
../bin/ICP: CMakeFiles/ICP.dir/src/ICP.cpp.o
../bin/ICP: CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o
../bin/ICP: CMakeFiles/ICP.dir/build.make
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
../bin/ICP: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
../bin/ICP: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
../bin/ICP: CMakeFiles/ICP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ypc/xiaoc/code/slam_module/ICP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/ICP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ICP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ICP.dir/build: ../bin/ICP

.PHONY : CMakeFiles/ICP.dir/build

CMakeFiles/ICP.dir/requires: CMakeFiles/ICP.dir/src/main.cpp.o.requires
CMakeFiles/ICP.dir/requires: CMakeFiles/ICP.dir/src/ICP.cpp.o.requires
CMakeFiles/ICP.dir/requires: CMakeFiles/ICP.dir/src/ICP_G2O.cpp.o.requires

.PHONY : CMakeFiles/ICP.dir/requires

CMakeFiles/ICP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ICP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ICP.dir/clean

CMakeFiles/ICP.dir/depend:
	cd /home/ypc/xiaoc/code/slam_module/ICP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ypc/xiaoc/code/slam_module/ICP /home/ypc/xiaoc/code/slam_module/ICP /home/ypc/xiaoc/code/slam_module/ICP/build /home/ypc/xiaoc/code/slam_module/ICP/build /home/ypc/xiaoc/code/slam_module/ICP/build/CMakeFiles/ICP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ICP.dir/depend
