# slam_module


1. [featureExtraction](https://github.com/yepeichu123/slam_module/tree/master/featureExtraction) 完成了利用OpenCV对图像进行特征提取的功能，特征点包含：ORB, SIFT, SURF, AKAZE, BRIEF。特征提取还包括两类方法，一类是直接提取，另一类是将原图细分为多个子图再提取。详情见featureExtraction/README.md；
   

2. [featureMatching](https://github.com/yepeichu123/slam_module/tree/master/featureMatching) 完成了利用OpenCV对步骤1中提取的特征点进行特征匹配的操作，匹配方法主要包括Flann和BruteForce匹配。Flann中主要用了最近邻比例法来筛选合适的匹配对，而暴力匹配法则是基于经验设置阈值条件来筛选。详情见featureMatching/README.md；
   

3. [triangularPoints](https://github.com/yepeichu123/slam_module/tree/master/triangularPoints) 完成了利用OpenCV以及Eigen对匹配的特征点对恢复空间点，重建环境的目标。ORB-SLAM和VINS-MONO均是通过构建 $Hx = 0$ 来计算空间点。三角化原理及其误差分析可以详见triangulatePoints/README.md;
   

4. [depthFilter](https://github.com/yepeichu123/slam_module/tree/master/depthFilter) 完成了利用多个开源SLAM系统使用的三角化方法，并计算其深度不确定性，利用不同帧对点进行深度滤波。深度滤波原理可以详见depthFilter/README.md;
   

5. [epipolarSearch](https://github.com/yepeichu123/slam_module/tree/master/epipolarConstrain) 完成了基于NCC的极线搜索方法，通过设置深度范围计算极线，利用NCC计算匹配分数。极线搜索可以详见epipolarSearch/README.md;
   

6. [stereoMatching](https://github.com/yepeichu123/slam_module/tree/master/stereoMatching) 完成了利用对齐图像的极线搜索计算NCC匹配分数，匹配分数符合阈值条件的，实现视差图和深度图。双目匹配可以详见stereoMatching/README.md;


7. [epipolarConstrain](https://github.com/yepeichu123/slam_module/tree/master/epipolarConstrain) 完成了利用对极几何约束求解两帧间相对运动，并验证求解误差。详情可见epipolarConstrain/README.md;


8. [ComputeHomography](https://github.com/yepeichu123/slam_module/tree/master/ComputeHomography) 完成了利用单应约束求解两帧间的单应矩阵，并将两个图像连接起来。详情可见ComputeHomography/README.md;


9. [PNP](https://github.com/yepeichu123/slam_module/tree/master/PNP) 完成了利用两帧（一个带三维点，一个带二维点）间的相对位姿估计，并验证求解误差。详情可见PNP/README.md;


10. [ICP](https://github.com/yepeichu123/slam_module/tree/master/ICP) 完成了利用两帧（均有三维点）间的相对位姿估计，并验证求解误差。详情可见ICP/README.md;


11. 
