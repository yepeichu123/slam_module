# EpipolarConstrain——Designed by 叶培楚

**算法流程**

1. 读取图像；
2. 读取相机内参；
3. 提取图像特征点；
4. 特征匹配；
5. 利用经验法进行滤除错误匹配；
6. （利用RANSAC进一步过滤错误匹配）；
7. 对极几何约束计算相对变换；
8. 验证结果。


### 参考文献

1. [使用cv::findFundamentalMat要注意的几点](http://blog.sina.com.cn/s/blog_4298002e01013w9a.html)
2. [对极几何、对极约束、单应性变换](https://blog.csdn.net/ak47fourier/article/details/82356771)
3. 