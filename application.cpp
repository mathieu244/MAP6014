/*
#################################################################################
#  Université: UQTR
#  Professeur: François Meunier
#  Cours: MAP6014
#  Création: Jessica Bélisle, Mathieu St-Yves
#################################################################################
*/
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
#include <unistd.h>

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

int parseInt(char* chars);

// ----------------------------------------------------------------------------------------

enum { CAP_MODE_BGR  = 0, // BGR24 (default)
       CAP_MODE_RGB  = 1, // RGB24
       CAP_MODE_GRAY = 2, // Y8
       CAP_MODE_YUYV = 3  // YUYV
     };

int main(int argc, char** argv) try
{

  //--------
  // Ouvrir la camera
  //--------
  VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

  if (!stream1.isOpened()) { //check if video device has been initialised
    cout << "Impossible d'ouvrir la caméra";
    return 1;
  }

  // Configurations de la résolution de la caméra
  //  stream1.set(CV_CAP_PROP_MODE, CV_CAP_MODE_YUYV);
  //  stream1.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
  //  stream1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);

  //--------
  // Chargement des réseaux de neuronnes pré-entrainé afin d'effectuer la détection de visage
  //--------

  // Face detector
  frontal_face_detector detector = get_frontal_face_detector();
  // Face landmarking model pour aligner les visages:  (see face_landmark_detection_ex.cpp for an introduction)
  shape_predictor sp;
  deserialize("../shape_predictor_5_face_landmarks.dat") >> sp;
  // DNN pour l'extraction du vecteur des points d'intérêt
  anet_type net;
  deserialize("../dlib_face_recognition_resnet_model_v1.dat") >> net;

  //--------
  // Définition des variables de cumul pendant la saisi et l'analyse
  //--------
  std::vector<matrix<float,0,1>> face_descriptors_cumul;
  std::vector<matrix<rgb_pixel>> faces_cumul;
  Mat cv_img;

  // Nombre de cluster affiché maximal (pour éviter le rafraichissement)
  std::vector<image_window> win_clusters(10);

  stream1.read(cv_img);
  // Cette ligne va échouer si la caméra retourne une image vide (camera inactive)
  dlib::cv_image<rgb_pixel> img(cv_img); //Conversion d'une image openCV en structure dlib

  //--------
  // Affichage de l'image à l'écran
  //--------
  image_window win(img);
  while (true) {

    //--------
    // Capture de l'image sur la webcam
    //--------
    stream1.read(cv_img);
    dlib::cv_image<rgb_pixel> img(cv_img); //Conversion d'une image openCV en structure dlib

    //--------
    // Affichage de l'image à l'écran
    //--------
    win.clear_overlay();
    win.set_image(img);

    //--------
    // Détection des visages et extraction du visage normalisé à une résolution de 150X150, normalisé et centré
    // Affiche une trace des visages trouvé
    //--------
    std::vector<matrix<rgb_pixel>> faces;

    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // Encadré pour l'apercu
        win.add_overlay(face);
    }
    if (faces.size() == 0)
    {
        cout << "Aucun visage sur l'image!" << endl;
    }else{
      //--------
      // Conversion des visage en vecteur 128D par le DNN
      // Chacun des vecteurs permet d'identifier un individu.
      // Lorsque 2 vecteurs sont proches, il s'agit de la même personne
      //--------
      std::vector<matrix<float,0,1>> face_descriptors = net(faces);
      faces_cumul.insert( faces_cumul.end(), faces.begin(), faces.end() );
      face_descriptors_cumul.insert( face_descriptors_cumul.end(), face_descriptors.begin(), face_descriptors.end() );

      //--------
      // Creation d'un graph pour regrouper les visages et calcul du nombre différente de personnes
      //--------
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
      const auto num_clusters = chinese_whispers(edges, labels); // Variante de l'algorithme de Markov-Chain-Clustering
      // On affiche le nombre total de personnes différente présente dans le cluster
      cout << "nombre de personne trouvée dans le cluster: "<< num_clusters << endl;

      // Affichage des visages dans leur cluster (fenêtre) respectif
      for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
      {
          std::vector<matrix<rgb_pixel>> temp;
          for (size_t j = 0; j < labels.size(); ++j)
          {
              if (cluster_id == labels[j])
                  temp.push_back(faces_cumul[j]);
          }
          win_clusters[cluster_id].set_title("Cluster #" + cast_to_string(cluster_id));
          win_clusters[cluster_id].set_image(tile_images(temp));
      }

      int microseconds = 50000;

      if (argc == 2)
      {
          microseconds = parseInt(argv[1]);
          waitKey(microseconds);
      }else{
        cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;
        cout << "appuyer sur enter pour continuer" << endl;
        cin.get();
      }
    }

  }

  cout << "appuyer sur enter pour terminer" << endl;
  cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

int parseInt(char* chars)
{
    int sum = 0;
    int len = strlen(chars);
    for (int x = 0; x < len; x++)
    {
        int n = chars[len - (x + 1)] - '0';
        sum = sum + pow(n, x);
    }
    return sum;
}
// ----------------------------------------------------------------------------------------
