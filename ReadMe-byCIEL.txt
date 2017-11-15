-----------------------------Function 1: CLMake3DFeature.m-----------------------------
一. 功能描述
将图像分割成超像素，并提取每个超像素块的Make3D特征

二. 文件位置
1.CLMake3DFeature.m
..\make3d\LearningCode\Debug\CLMake3DFeature.m

2. 输入图片（.jpg）及其标注信息（.txt）的位置
..\make3d\LightImages

三. 使用说明
step1. 将文件夹修改为：..\make3d\LearningCode
step2. 初始化路径
	   在命令行窗口中输入：InitialPath(true)   
step3. 输入：CLMake3DFeature('../outputTest/test.jpg','../OutputTest/')

四. 输出特征文件
1. 位置
..\make3d\OutputTest
2. 文件说明
（1）每张图片对应一个.txt文件，文件名与图片名相同
（2）第1列：超像素的索引序号
     第2列：0或1，0表示该超像素不是光源，1表示是光源
	 第3-4列：超像素的中心坐标（y,x）
	 第5-end列：
	 分成5个部分（超像素+四邻域）：该超像素，左边的超像素，上，右，下
	 每部分有68列，17*2*2=68,其中，17是滤波器输出；第一个2是k取2和4的情况；第二个2是两种尺度（100%，50%）
	 
五. 操作系统
win10 64bit，MATLAB2015b