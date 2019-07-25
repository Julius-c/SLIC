# <center>Simple Linear Iterative Clustering</center>

## <center>Cheng Chen</center>

### <center>Overview</center>

超像素指一系列位置相近且颜色, 纹理, 特征等特征相似的像素点组成的小区域. 这些小区域大多保留了进一步进行图像分割的有效信息, 且一般不会破坏图像中物体的边界信息.

超像素分割即将一副像素级的输入图片, 分割成区域级的图, 是对基本信息元素进行的抽象.

### <center>Algorithm Description</center>

1. 算法输入:原始图像、预想分割的超像素数$k$、簇内最大颜色距离$N_c$

2. 算法步骤:

   1. 根据图像大小及预想分割的超像素数计算簇中心的步长$S$, 亦即簇内最大空间距离$N_s$

      $$S = N_c = \sqrt{(ImageHeight * ImageWidth / k)}​$$

   2. 将图像从$RGB$空间转化到$LAB$空间, 每个像素点对应一个五维向量$[l, a, b, x, y]$

   3. 根据步长$S$在图像上规则选取$k$个簇中心

   4. 将每个簇中心移动到其$3\times 3​$邻域里梯度最小(最平坦)的点

   5. 初始化所有像素点所在的簇$label$为$-1$, 到簇中心的**距离**$distance$为$\infty$

   6. 迭代如下过程:

      1. 对每个簇中心$2S\times 2S$的邻域内的点, 计算其到簇中心的**距离**

         颜色距离为$d_c=\sqrt{(l_j-l_i)^2+(a_j-a_i)^2+(b_j-b_i)^2}​$

         空间距离为$d_s=\sqrt{(x_j-x_i)^2+(y_j-y_i)^2}$

         根据$N_c, N_s$正则化距离为$distance = \sqrt{(\frac{d_c}{N_c})^2+(\frac{d_s}{N_s})^2}​$

      2. 根据到簇中心距离将像素划分进距离最近的簇

      3. 将新的簇内像素进行平均计算得到新的簇中心

3. 算法输出: 将处于簇边界的像素点标注在图像中加以显示

### <center>Algorithm Analysis</center>

1. 初始化时在$3\times 3$邻域内重新选取梯度最小的点, 是为了避免初始簇中心像素点落在梯度较大的边界上

2. 与$k​$均值算法不同的是, $SLIC​$算法不计算簇中心到所有样本点的距离, 仅计算到其$2S\times 2S​$邻域内的像素点的距离, 加速算法收敛

   > $kmeans$算法又名​$k$均值算法. 其**算法思想**大致为: 先从样本集中随机选取​$k$个样本作为**簇中心**, 并计算所有样本与这​$k​$个“簇中心”的距离, 对于每一个样本, 将其划分到与其**距离最近**的“簇中心”所在的簇中, 对于新的簇计算各个簇的新的“簇中心”.

3. 实践发现$10​$次迭代对绝大部分图片都可以得到较理想效果, 故一般迭代次数取$10​$

4. 最大颜色距离$N_c​$随图片不同而不同, 也随聚类不同而不同, 故一般取一固定常数$m\in[1, 40]​$

### <center>Results</center>

<div align="middle">
  <table style="width=100%">
	  <tr>
  		<td>
    	  <img src="images/input.jpg" align="middle" width="400px"/>
    		<figcaption align="middle">Input Image</figcaption>
  		</td>
  		<td>
    	  <img src="images/output.jpg" align="middle" width="400px"/>
    		<figcaption align="middle">SuperPixels Segmentation</figcaption>
  		</td>
	  </tr>
  </table>
</div>

### <center>Command Line</center>

`python slic.py IMAGE SUPERPIXELS NC`