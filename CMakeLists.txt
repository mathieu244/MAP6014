#################################################################################
#  Université: UQTR
#  Professeur: François Meunier
#  Cours: MAP6014
#  Création: Jessica Bélisle, Mathieu St-Yves
#################################################################################

cmake_minimum_required(VERSION 2.8.12)
project(FaceRecognition) # Nom du projet
add_subdirectory(dlib dlib_build)
# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#

#################################################################################
#  A CMakeLists.txt file can compile more than just one program.  So below we
#  tell it to compile the other dlib example programs using pretty much the
#  same CMake commands we used above.
#################################################################################


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so the build fail.")
else()
   find_package(OpenCV QUIET)
   find_package(dlib REQUIRED)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      add_executable(application application.cpp)
      target_link_libraries(application dlib::dlib ${OpenCV_LIBS})
   else()
      message("OpenCV not found, so we won't build the application.")
   endif()
endif()
