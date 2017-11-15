//$$ oneShot3d.h 
#include <math.h>

#ifdef use_namespace
namespace RBD_COMMON {
#endif

class DefaultParaValues 
{

public:
   void setFolderNames(char *taskName,
                  char *DepthPara,
                  char *VarPara,
                  char *GroundSkyPara,
                  bool ScratchFlag,
                  char *SFeaPara,
                  char *FeaPara);

//% ============== Highly changealbe parameters ========================
   int SegVertYSize = 1200;
   int SegHoriXSize = 900;
   int VertYNuPatch = 55;
   int HoriXNuPatch = 61;//%305;%61;
   int VertYNuDepth = 55;
   int HoriXNuDepth = 305;
   int PopUpHoriX = 600;
   int PopUpVertY = 800;
   int batchSize = 10;
   int NuRow_default = 55;
   int WeiBatchSize = 5;
   int TrainVerYSize = 2272;
   int TrainHoriXSize = 1704;
   int MempryFactor =2;
// % pics info
   double Horizon = 0.5;// % the position of the horizon in a pics (the bottom of the pic is 0 top is 1 middle is 1/2)
// % segmentation info
   double sigm = 0.5;//%0.8%0.3;
   int k = 100; //%300;%200;
   int minp = 100; //%150;%20;

// %================ camera info from kyle's code
// % This can probably also be estimated from jpeg header
   double fx = 2400.2091651084;
   double fy = 7.3312729885838;
   double Oy = 1110.7122391785729; //%2272/2; %
   double Ox = 833.72104535435108; //%1704/2; %
   double a = 1704/fy; //%0.70783777; %0.129; % horizontal physical size of image plane normalized to focal length (in meter)
   double b = 2272/fx; //%0.946584169;%0.085; % vertical physical size of image plane normalized to focal length (in meter)
   double Ox = 1-(Ox/1704); //%0.489272914; % camera origin offset from the image center in horizontal direction
   double Oy = 1-(Oy/2272); //%0.488886982; % camera origin offset from the image center in vertical direction

};

#ifdef use_namespace
}
#endif

