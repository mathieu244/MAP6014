#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
  //--------
  // Ouvrir la camera
  //--------
  VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

  if (!stream1.isOpened()) { //check if video device has been initialised
    cout << "cannot open camera";
    return 1;
  }

  //--------
  // Chargement des réseaux de neuronnes pré-entrainé afin d'effectuer la détection de visage
  //--------

  // The first thing we are going to do is load all our models.  First, since we need to
  // find faces in the image we will need a face detector:
  frontal_face_detector detector = get_frontal_face_detector();
  // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
  shape_predictor sp;
  deserialize("../shape_predictor_5_face_landmarks.dat") >> sp;
  // And finally we load the DNN responsible for face recognition.
  anet_type net;
  deserialize("../dlib_face_recognition_resnet_model_v1.dat") >> net;


  //--------
  // Définition des variables de cumul pendant la saisi et l'analyse
  //--------
  std::vector<matrix<float,0,1>> face_descriptors_cumul;
  std::vector<matrix<rgb_pixel>> faces_cumul;
  matrix<rgb_pixel> img;
  Mat cv_img;

  while (true) {

    //--------
    // Capture de l'image sur la webcam
    //--------


    stream1.read(cv_img);
    dlib::cv_image<rgb_pixel> img(cv_img); //Conversion d'une image openCV en structure dlib

    //--------
    // Affichage de l'image à l'écran
    //--------
    image_window win(img);

    //--------
    // Détection des visages et extraction du visage normalisé à une résolution de 150X150
    // Affiche une trace des visages trouvé
    //--------

    // Run the face detector on the image of our action heroes, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.
        win.add_overlay(face);
    }

    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        //return 1;
    }else{
      //--------
      // Conversion des visage en vecteur 128D
      // Cumul des vecteurs pour une reconnaissance globale
      //--------

      // This call asks the DNN to convert each face image in faces into a 128D vector.
      // In this 128D vector space, images from the same person will be close to each other
      // but vectors from different people will be far apart.  So we can use these vectors to
      // identify if a pair of images are from the same person or from different people.
      std::vector<matrix<float,0,1>> face_descriptors = net(faces);
      faces_cumul.insert( faces_cumul.end(), faces.begin(), faces.end() );
      face_descriptors_cumul.insert( face_descriptors_cumul.end(), face_descriptors.begin(), face_descriptors.end() );

      //--------
      // Creation d'un graph pour regrouper les visages
      //--------

      // In particular, one simple thing we can do is face clustering.  This next bit of code
      // creates a graph of connected faces and then uses the Chinese whispers graph clustering
      // algorithm to identify how many people there are and which faces belong to whom.
      std::vector<sample_pair> edges;
      for (size_t i = 0; i < face_descriptors_cumul.size(); ++i)
      {
          for (size_t j = i; j < face_descriptors_cumul.size(); ++j)
          {
              // Faces are connected in the graph if they are close enough.  Here we check if
              // the distance between two face descriptors is less than 0.6, which is the
              // decision threshold the network was trained to use.  Although you can
              // certainly use any other threshold you find useful.
              if (length(face_descriptors_cumul[i]-face_descriptors_cumul[j]) < 0.6)
                  edges.push_back(sample_pair(i,j));
          }
      }
      std::vector<unsigned long> labels;
      const auto num_clusters = chinese_whispers(edges, labels);
      // This will correctly indicate that there are 4 people in the image.
      cout << "number of people found in the image: "<< num_clusters << endl;

      // Now let's display the face clustering results on the screen.  You will see that it
      // correctly grouped all the faces.
      std::vector<image_window> win_clusters(num_clusters);
      for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
      {
          std::vector<matrix<rgb_pixel>> temp;
          for (size_t j = 0; j < labels.size(); ++j)
          {
              if (cluster_id == labels[j])
                  temp.push_back(faces_cumul[j]);
          }
          win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
          win_clusters[cluster_id].set_image(tile_images(temp));
      }
      cout << "hit enter refresh" << endl;
      cin.get();
    }

  }


  // Finally, let's print one of the face descriptors to the screen.
  //cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

  // It should also be noted that face recognition accuracy can be improved if jittering
  // is used when creating face descriptors.  In particular, to get 99.38% on the LFW
  // benchmark you need to use the jitter_image() routine to compute the descriptors,
  // like so:
  //matrix<float,0,1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
  //cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
  // If you use the model without jittering, as we did when clustering the bald guys, it
  // gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
  // procedure a little more accurate but makes face descriptor calculation slower.


  cout << "hit enter to terminate" << endl;
  cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------
