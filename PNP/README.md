# PNP——Designed by 叶培楚

**算法流程**

1. 读取图像；

2. 读取相机内参和尺度因子；

3. 提取图像特征点；

4. 特征匹配并过滤错误匹配；

5. 从深度图为参考帧匹配的特征点提取深度值，并计算三维坐标；

6. 利用PNP计算相对姿态；（有三种策略可以选择，一种是直接利用OpenCV的PNPRansac方法，另一种是利用图优化的方法进行优化，最后一种是利用OpenCV的PNPRansac函数获取相对姿态初值，再利用图优化优化该姿态）

7. 误差估计。


**算法概述**

在本次代码中，笔者首先用OpenCV提供的solvePnPRansac(...)函数实现了PNP的基本功能，估算出了两帧间的相对运动，并利用RANSAC方法找出了内点。

RANSAC方法可以有效地找出满足相对姿态的最大内点集。通过随机提取几组点，估算出相对姿态，将所有点进行重投影验证相对姿态，统计内点数量。以此找出一组满足有最大内点数的相对姿态。

笔者利用g2o实现了一个非线性优化版本的PNP，通过构建非线性函数，以单位阵和零位移为初值，构建二元边（或一元边）生成图优化问题，进行优化。最终结果也是正确的。

因为非线性优化比较依赖于初值条件，假如初值与期望值差别较大，很容易得到错误解。因此我们通过将上述两种方法结合，先利用OpenCV的方法估算出相对姿态初值，再利用非线性优化对相对位姿进行优化。


PS： emmm... 在实现优化部分的代码时，因为对g2o不太熟悉，笔者花了一点时间。不过经验还是来源于一点一滴的积累。只要你付出过努力，收获总是会有的。


### PNP算法

&ensp; &ensp; 在视觉SLAM中，我们通常会有不同的观测数据。

比如单目相机中，我们获取的两帧图像通常只有匹配的像素坐标，这时我们就只能利用[对极几何](https://www.cnblogs.com/yepeichu/p/12604678.html)来优化相对位姿，如果我们的特征点分布大体上接近一个平面，那么我们还可以估算两帧间的[单应矩阵](https://www.cnblogs.com/yepeichu/p/12612273.html)；

如果我们有RGB-D相机，我们还可以获得匹配像素对应的深度信息，那么我们就得到了两组三维匹配点，这时就可以通过[ICP（迭代最近点）](https://www.cnblogs.com/yepeichu/p/12632767.html)来进行运动估计，ICP又分利用SVD分解和Bundle Adjustment两种方式，但是本质上并无区别，毕竟SVD也是构建非线性最小二乘问题。不过，Bundle Adjustment问题可以同时优化点和位姿。

如果我们有一组三维点匹配二维点，这种类型的数据获取方式有多种。典型的是可以通过RBG-D来获取，前面我们提到RGB-D采集的数据可以用ICP进行优化位姿，实际上由于RGB-D相机本身的噪声，深度值估计是有误差的。因此在SLAM中通常会用PNP的方法来做估计，PNP也就是我们本讲要介绍的内容，利用一个带噪声的深度值进行位姿估计，总比用两个都带噪声的深度值来得合适些。另一种渠道是，单目三角化得到的地图点重投影到新帧中构建的PNP问题，原理一致。此外，还有双目图像。


#### 已知条件

&ensp; &ensp; 匹配的 $n$ 组三维点-二维点：
$$P^{r} = \{P_{1}^{r}, P_{2}^{r}, \dots, P_{n}^{r}\}, p^{c} = \{p_{1}^{c}, p_{2}^{c}, \dots, p_{n}^{c}\}$$

其中，$P^{r}$ 表示参考帧中的三维点，$p^{c}$ 表示当前帧匹配的二维点（像素坐标）。数据获取方式如前面介绍的。

#### 问题

&ensp; &ensp; 在已知条件下，求解参考帧到当前帧的相对位姿。（旋转矩阵$R$ 和位移向量$t$）

#### 方法一：直接线性变换（DLR）

&ensp; &ensp; 直接线性变换与我们前面介绍的对极几何，单应矩阵的计算是类似的，都是忽略相对位姿本身的性质，直接将其视为一个数值矩阵，优化完才利用某些约束条件恢复相对运动。

&ensp; &ensp; 假设我们有一对匹配对：$P_{i}^{r} = [X, Y, Z, 1]^{T}$ 和 $p_{i}^{c} = [u, v, 1]^{T}$，两个坐标均为归一化坐标。假设相对位姿为：$T = [R | t]$，$p_{i}^{r}$ 的深度值为 $s$，则我们有：
$$
s
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix} = \begin{bmatrix} t_{1} & t_{2} & t_{3} & t_{4} \\
                                t_{5} & t_{6} & t_{7} & t_{8} \\
                                t_{9} & t_{10} & t_{11} & t_{12} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$


通过令 $t = \begin{bmatrix} t_{1}^{T} \\ t_{2}^{T} \\ t_{3}^{T} \end{bmatrix}$，则上式可以变成：
$$
s
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix} = \begin{bmatrix} t_{1}^{T} \\
                                t_{2}^{T} \\
                                t_{3}^{T} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} t_{1}^{T} \\
                                t_{2}^{T} \\
                                t_{3}^{T} \end{bmatrix} P_{i}^{c}
$$


可以得到：
$$
\begin{aligned}
u = \frac{t_{1}^{T}P_{i}^{c}}{t_{3}^{T}P_{i}^{c}} \\
v = \frac{t_{2}^{T}P_{i}^{c}}{t_{3}^{T}P_{i}^{c}}
\end{aligned}
$$

通过上述式子，我们去除了尺度因子的约束。进一步简化，我们得到两个约束条件：
$$
\begin{aligned}
-t_{3}^{T} u P_{i}^{c} + t_{1}^{T}P_{i}^{c} = 0 \Rightarrow \begin{bmatrix}P_{i}^{c} & 0 & -uP_{i}^{c} \end{bmatrix} \begin{bmatrix} t_{1}^{T} \\ t_{2}^{T} \\ t_{3}^{T}  \end{bmatrix}	 \\
-t_{3}^{T} v P_{i}^{c} + t_{2}^{T}P_{i}^{c} = 0 \Rightarrow \begin{bmatrix}0 & P_{i}^{c} & -vP_{i}^{c} \end{bmatrix} \begin{bmatrix} t_{1}^{T} \\ t_{2}^{T} \\ t_{3}^{T}  \end{bmatrix}	 \\
\end{aligned}
$$

由于相对位姿共有 $12$ 个未知数，因此至少需要六组点，才能提供 $12$ 个约束条件。

#### 参考资料

1. [g2o定义边](https://blog.csdn.net/weixin_42905141/article/details/100830126)
2. [SLAM之李群李代数工具](http://www.mathsword.com/slam_se3_so3/)
3. [李代数求导](http://www.mathsword.com/diff_se3_so3/)
4. [李代数的导数](https://blog.csdn.net/qq_40007147/article/details/103349460)
5. [相机位姿求解问题？](https://www.zhihu.com/question/51510464)
6. [3D-2D相机位姿估计](https://www.jianshu.com/p/f16e5b5cc47d) 
7. [特征法前端（二）](https://zhuanlan.zhihu.com/p/35519429)
8. 