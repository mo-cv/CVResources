# CVResources
## 通用库/General Library
* ==OpenCV==  
无需多言。
* RAVL
Recognition And Vision Library. 线程安全。强大的IO机制。包含AAM。
* CImg
很酷的一个图像处理包。整个库只有一个头文件。包含一个基于PDE的光流算法。
## 图像，视频IO/Image, Video IO
### FreeImage
### DevIL
### ImageMagick
### FFMPEG
### VideoInput
### portVideo
AR相关/Augmented Reality
ARToolKit
基于Marker的AR库
ARToolKitPlus
ARToolKit的增强版。实现了更好的姿态估计算法。
PTAM
实时的跟踪、SLAM、AR库。无需Marker，模板，内置传感器等。
BazAR
基于特征点检测和识别的AR库。
局部不变特征/Local Invariant Feature
VLFeat
目前最好的Sift开源实现。同时包含了KD-tree，KD-Forest，BoW实现。
Ferns
基于Naive Bayesian Bundle的特征点识别。高速，但占用内存高。
SIFT By Rob Hess
基于OpenCV的Sift实现。
目标检测/Object Detection
AdaBoost By JianXin.Wu
又一个AdaBoost实现。训练速度快。
行人检测 By JianXin.Wu
基于Centrist和Linear SVM的快速行人检测。
（近似）最近邻/ANN
FLANN
目前最完整的（近似）最近邻开源库。不但实现了一系列查找算法，还包含了一种自动选取最快算法的机制。
ANN
另外一个近似最近邻库。
SLAM & SFM
SceneLib [LGPL]
monoSLAM库。由Androw Davison开发。
图像分割/Segmentation
SLIC Super Pixel
使用Simple Linear Iterative Clustering产生指定数目，近似均匀分布的Super Pixel。
目标跟踪/Tracking
TLD
基于Online Random Forest的目标跟踪算法。
KLT
Kanade-Lucas-Tracker
Online boosting trackers
Online Boosting Trackers
直线检测/Line Detection
DSCC
基于联通域连接的直线检测算法。
LSD [GPL]
基于梯度的，局部直线段检测算子。
指纹/Finger Print
pHash [GPL]
基于感知的多媒体文件Hash算法。（提取，对比图像、视频、音频的指纹）
视觉显著性/Visual Salience
Global Contrast Based Salient Region Detection
Ming-Ming Cheng的视觉显著性算法。
FFT/DWT
FFTW [GPL]
最快，最好的开源FFT。
FFTReal [WTFPL]
轻量级的FFT实现。许可证是亮点。
音频处理/Audio processing
STK [Free]
音频处理，音频合成。
libsndfile [LGPL]
音频文件IO。
libsamplerate [GPL ]
音频重采样。
小波变换
快速小波变换（FWT）
FWT
BRIEF: Binary Robust Independent Elementary Feature 一个很好的局部特征描述子，里面有FAST corner + BRIEF实现特征点匹配的DEMO：http://cvlab.epfl.ch/software/brief/
http://code.google.com/p/javacv

Java打包的OpenCV, FFmpeg, libdc1394, PGR FlyCapture, OpenKinect, videoInput, and ARToolKitPlus库。可以放在Android上用~
 
libHIK,HIK SVM，计算HIK SVM跟Centrist的Lib。http://c2inet.sce.ntu.edu.sg/Jianxin/projects/libHIK/libHIK.htm
 
一组视觉显著性检测代码的链接：http://cg.cs.tsinghua.edu.cn/people/~cmm/saliency/


介绍n款计算机视觉库/人脸识别开源库/软件
计算机视觉库 OpenCV
OpenCV是Intel®开源计算机视觉库。它由一系列 C 函数和少量 C++ 类构成，实现了图像处理和计算机视觉方面的很多通用算法。 OpenCV 拥有包括 300 多个C函数的跨平台的中、高层 API。它不依赖于其它的外部库——尽管也可以使用某些外部库。 OpenCV 对非商业...
人脸识别 faceservice.cgi
faceservice.cgi 是一个用来进行人脸识别的 CGI 程序， 你可以通过上传图像，然后该程序即告诉你人脸的大概坐标位置。faceservice是采用 OpenCV 库进行开发的。
OpenCV的.NET版 OpenCVDotNet
OpenCVDotNet 是一个 .NET 对 OpenCV 包的封装。
人脸检测算法 jViolajones
jViolajones是人脸检测算法Viola-Jones的一个Java实现，并能够加载OpenCV XML文件。 示例代码：http://www.oschina.net/code/snippet_12_2033
Java视觉处理库 JavaCV
JavaCV 提供了在计算机视觉领域的封装库，包括：OpenCV、ARToolKitPlus、libdc1394 2.x 、PGR FlyCapture和FFmpeg。此外，该工具可以很容易地使用Java平台的功能。 JavaCV还带有硬件加速的全屏幕图像显示（CanvasFrame），易于在多个内核中执行并行代码（并...
运动检测程序 QMotion
QMotion 是一个采用 OpenCV 开发的运动检测程序，基于 QT。
视频监控系统 OpenVSS
OpenVSS - 开放平台的视频监控系统 - 是一个系统级别的视频监控软件视频分析框架（VAF）的视频分析与检索和播放服务，记录和索引技术。它被设计成插件式的支持多摄像头平台，多分析仪模块（OpenCV的集成），以及多核心架构。
手势识别 hand-gesture-detection
手势识别，用OpenCV实现
人脸检测识别 mcvai-tracking
提供人脸检测、识别与检测特定人脸的功能，示例代码 cvReleaseImage( &gray ); cvReleaseMemStorage(&storage); cvReleaseHaarClassifierCascade(&cascade);...
人脸检测与跟踪库 asmlibrary
Active Shape Model Library (ASMLibrary©) SDK, 用OpenCV开发，用于人脸检测与跟踪。
Lua视觉开发库 libecv
ECV 是 lua 的计算机视觉开发库(目前只提供Linux支持)
OpenCV的.Net封装 OpenCVSharp
OpenCVSharp 是一个OpenCV的.Net wrapper，应用最新的OpenCV库开发，使用习惯比EmguCV更接近原始的OpenCV，有详细的使用样例供参考。
3D视觉库 fvision2010
基于OpenCV构建的图像处理和3D视觉库。 示例代码： ImageSequenceReaderFactory factory; ImageSequenceReader* reader = factory.pathRegex("c:/a/im_%03d.jpg", 0, 20); //ImageSequenceReader* reader = factory.avi("a.avi"); if (reader == NULL) { ...
基于QT的计算机视觉库 QVision
基于 QT 的面向对象的多平台计算机视觉库。可以方便的创建图形化应用程序，算法库主要从 OpenCV，GSL，CGAL，IPP，Octave 等高性能库借鉴而来。
图像特征提取 cvBlob
cvBlob 是计算机视觉应用中在二值图像里寻找连通域的库.能够执行连通域分析与特征提取.
实时图像/视频处理滤波开发包 GShow
GShow is a real-time image/video processing filter development kit. It successfully integrates DirectX11 with DirectShow framework. So it has the following features: GShow 是实时 图像/视频 处理滤波开发包，集成DiretX11。...
视频捕获 API VideoMan
VideoMan 提供一组视频捕获 API 。支持多种视频流同时输入（视频传输线、USB摄像头和视频文件等）。能利用 OpenGL 对输入进行处理，方便的与 OpenCV，CUDA 等集成开发计算机视觉系统。
开放模式识别项目 OpenPR
Pattern Recognition project（开放模式识别项目），致力于开发出一套包含图像处理、计算机视觉、自然语言处理、模式识别、机器学习和相关领域算法的函数库。
OpenCV的Python封装 pyopencv
OpenCV的Python封装，主要特性包括： 提供与OpenCV 2.x中最新的C++接口极为相似的Python接口，并且包括C++中不包括的C接口 提供对OpenCV 2.x中所有主要部件的绑定：CxCORE (almost complete), CxFLANN (complete), Cv (complete), CvAux (C++ part almost...
视觉快速开发平台 qcv
计算机视觉快速开发平台，提供测试框架，使开发者可以专注于算法研究。
图像捕获 libv4l2cam
对函数库v412的封装，从网络摄像头等硬件获得图像数据，支持YUYV裸数据输出和BGR24的OpenCV  IplImage输出
计算机视觉算法 OpenVIDIA
OpenVIDIA projects implement computer vision algorithms running on on graphics hardware such as single or multiple graphics processing units(GPUs) using OpenGL, Cg and CUDA-C. Some samples will soon support OpenCL and Direct Compute API&apos;...
高斯模型点集配准算法 gmmreg
实现了基于混合高斯模型的点集配准算法，该算法描述在论文： A Robust Algorithm for Point Set Registration Using Mixture of Gaussians, Bing Jian and Baba C. Vemuri. ，实现了C++/Matlab/Python接口...
模式识别和视觉库 RAVL
Recognition And Vision Library (RAVL) 是一个通用 C++ 库，包含计算机视觉、模式识别等模块。
图像处理和计算机视觉常用算法库 LTI-Lib
LTI-Lib 是一个包含图像处理和计算机视觉常用算法和数据结构的面向对象库，提供 Windows 下的 VC 版本和 Linux 下的 gcc 版本，主要包含以下几方面内容： 1、线性代数 2、聚类分析 3、图像处理 4、可视化和绘图工具
OpenCV优化 opencv-dsp-acceleration
优化了OpenCV库在DSP上的速度。
C++计算机视觉库 Integrating Vision Toolkit
Integrating Vision Toolkit (IVT) 是一个强大而迅速的C++计算机视觉库，拥有易用的接口和面向对象的架构，并且含有自己的一套跨平台GUI组件，另外可以选择集成OpenCV
计算机视觉和机器人技术的工具包 EGT
The Epipolar Geometry Toolbox (EGT) is a toolbox designed for Matlab (by Mathworks Inc.). EGT provides a wide set of functions to approach computer vision and robotics problems with single and multiple views, and with different vision se...
OpenCV的扩展库 ImageNets
ImageNets 是对OpenCV 的扩展，提供对机器人视觉算法方面友好的支持，使用Nokia的QT编写界面。
libvideogfx
视频处理、计算机视觉和计算机图形学的快速开发库。
Matlab计算机视觉包 mVision
Matlab 的计算机视觉包，包含用于观察结果的 GUI 组件，貌似也停止开发了，拿来做学习用挺不错的。
Scilab的计算机视觉库 SIP
SIP 是 Scilab（一种免费的类Matlab编程环境）的图像处理和计算机视觉库。SIP 可以读写 JPEG/PNG/BMP 格式的图片。具备图像滤波、分割、边缘检测、形态学处理和形状分析等功能。
STAIR Vision Library
STAIR Vision Library (SVL) 最初是为支持斯坦福智能机器人设计的，提供对计算机视觉、机器学习和概率统计模
