# featureExtraction——Designed by 叶培楚

通过FeatureExtraction类将特征提取的内容封装起来。

特征类型包括：
1. ORB
2. SIFT
3. SURF
4. AKAZE
5. BRIEF
PS: 想要使用SIFT, SURF和BRIEF需要安装Opencv-contrib包。

特征提取函数有两种类型：
1. 利用Opencv函数直接对原图像进行特征提取；
2. 对原图像进行简单的分割，之后对每个子图进行特征提取，再将所有特征点汇总。
PS: 第二种方法相对于ORB-SLAM在提取的结果上，几乎没有可比性。毕竟ORB-SLAM是通过对图像进行非常细致的分割，再利用FAST来检测角点，从头到尾都是自己手撸的。我这边是用了OpenCV开源库的函数。若有需要，可以基于此进行拓展。
