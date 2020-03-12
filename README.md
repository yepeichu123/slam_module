# slam_module

1. **featureExtraction**完成了利用OpenCV对图像进行特征提取的功能，特征点包含：ORB, SIFT, SURF, AKAZE, BRIEF。特征提取还包括两类方法，一类是直接提取，另一类是将原图细分为多个子图再提取。详情见featureExtraction/README.md；
   
2. **featureMatching**完成了利用OpenCV对步骤1中提取的特征点进行特征匹配的操作，匹配方法主要包括Flann和BruteForce匹配。Flann中主要用了最近邻比例法来筛选合适的匹配对，而暴力匹配法则是基于经验设置阈值条件来筛选。详情见featureMatching/README.md；

3. 