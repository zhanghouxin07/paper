# 10.14
## 牛顿迭代法 
牛顿迭代法广泛应用于计算机编程中。

```math
x_1 = x_0 - \frac{{f(x_0)}}{{f'\left( {{x_0}} \right)}}
```
称为f(x_0)的一阶近似。

## Motion Planning Networks
> Qureshi, Ahmed H., et al. "Motion planning networks." 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019.

a Deep Neural Network (DNN) based iterative motion planning algorithm, called MPNet (Motion Planning Networks) 来解决高维空间问题，包括
- an encoder network： 

The encoder network learns to encode a point cloud of the obstacles into a latent space. 
- a planning network

The planning network learns to predict the robot configuration at time step t + 1 given the robot configuration at time t.

另一个方法，As neural networks do not provide theoretical guarantees on their performance. Therefore, we also propose a hybrid algorithm which combines MPNet with any existing classical planning algorithm. 

**Problem Formulation**:
From the initial state to the goal state, to solve a soluation path.

**MPNET:ANEURAL MOTION PLANNER**:

*Encoder-Decoder*

(编码-解码)是深度学习中常见的模型，例如无监督模型的auto-encoding，神经网络机器翻译。最显著的特征是“end to end”的算法，其实就是Sequence to sequence learning。
所谓编码就是将输入序列转化成一个固定长度的向量；
解码就是将之前生成的向量在转化成输出序列。

encoder部分是将输入序列表示成一个带有语义的向量，decoder部分是以encoder生成的hidden state vector作为输入“解码”出目标文本序列；

局限性：编码和解码之间的唯一联系就是一个固定长度的语义向量C。一是语义向量无法完整表示整个序列的信息，先输入的信息会被后面的信息稀释掉。

*Attention 模型*

这种模型在产生输出的时候，还会产生一个“注意力范围”表示接下来输出的时候要重点关注输入序列中的哪些部分，然后根据关注的区域来产生下一个输出，如此往复。

利用权重，对特征进行加权求和，权值越大，对当前识别的贡献就越大。将encoder的特征以更好的方式呈现给decoder。

**constituent function**

- Enet

障碍物点云加入到潜在空间中（定位障碍物）

- Pnet：

前馈神经网络 and drop out ∈[0,1]。 
根据目标点，起始点。基于当前位置predict下一个位置。

- Lazy States Contraction

连接非连续状态并移除lazy state（waypoint）

- Steering

检查两个连续的离散点间的路径是否是collision-free的。

- isFeasible

检查整个路径是否是 collision-free的。

- neural planner

Generate bidirectional paths by swap function. The path is extended from start-points to the end-points, which is heuristic greedy and fast.

- Replanning

如果给定一个轨迹，则检查轨迹是否是无碰撞的。
若某两点之间是 not connectable的。则根据以下replanning方法：
1. 通过神经网络的方法重新规划，即前文的方法。
2. Hybrid method：若神经网络的方法没有生成一条可用的路径，则使用经典的运动控制（路径规划）算法来规划路径。

**Implementation Details**


This section gives the implementation details of MPNet,
for additional details refer to supplementary material. The
proposed neural models were implemented in PyTorch [19].
For environments other than Baxter, the benchmark methods,
Informed-RRT* and BIT*, were implemented in Python, and
their times were compared against the CPU-time of MPNet.
The Baxter environments were implemented with MoveIt!
[20] and ROS. In these environments, we use a C++ OMPL
[21] implementation of BIT* to compare against a C++
implementation of MPNet. The system used for training and
testing has 3.40GHz× 8 Intel Core i7 processor with 32 GB
RAM and GeForce GTX 1080 GPU. The remaining section
explains different modules that lead to MPNet.
- Data collection

We generate 110 different workspaces for each presented
case, i.e., simple 2D (s2D), rigid-body (rigid), complex
2D (c2D) and 3D (c3D). In each of the workspaces,
5000 collision-free, near-optimal, paths were generated using
RRT*. The training dataset comprised of 100 workspaces
with 4000 paths in each workspace. For testing, two types
of test datasets were created to evaluate the proposed and
benchmark methods. The first test dataset, seen-Xobs, comprised of already seen 100 workspaces with 200 unseen start
and goal configurations in each workspace. The second test
dataset, unseen-Xobs, comprised of completely unseen 10
workspaces where each contained 2000 unseen start and
goal configurations. In the Baxter experiments, we created
a dataset comprised of ten challenging simulated environments, and we show the execution on the real robot. For each
environment, we collected 900 paths for training and 100
paths for testing. The obstacle point clouds were obtained
using a Kinect depth camera with the PCL [22] and pcl ros2
package.
- Models Architecture

\
**REsults**：
- 比较了不同方法的路径效果，运行时间。（nr，hr，rrt*，bit*）
- 效果分析
- 计算复杂度Computational complexity分析

**end**

---

# 10.15
## 单词
insight 洞察，洞悉，
avert 转移

## 第12章 应用
### 大规模深度学习
大规模深度学习的基本思想基于联结主义，尽管单个神经元没法表现出智能。但是大量的神经元或特征聚集在一起往往能表现出智能。

现代神经网络的实现基于图形处理器（graphics processing unit）gpu。gpu的并行计算处理特点加速了神经网络的训练（参数更新，激活值，梯度值占据了大量缓冲区）。神经网络从gpu中获益匪浅。
然而使用gpu进行更快速的计算不是一个容易的事情，多线程的计算意味着跟复杂的架构。NVIDIA的cuda语言最终成为了一个库来加速计算。

大规模分布式实现：异步随机梯度下降法中几个处理器的核共用参数内存。

模型压缩：使用一个更小的模型，来减少内存使用和运行时间。

动态结构：使用级联的分类器，按顺序执行分类器。使用更少的计算拒绝不包含对象的输入；二是‘开关’：其中隐藏单元可以从不同的神经元接受输入。但是在大规模应用中没有被证明是有效的。
### 计算机视觉
- 预处理：预处理可以增强模型的泛化能力，应用于训练机、数据集。对比度是能够安全移除的最明显的变化源之一；
全局对比度归一化和局部对比度归一化两种方法。lcn确保对比度在每个小窗口上被归一化，从而使模型避免除以0的情况。
- 数据集增强：随机转换和旋转可以增强数据集，例如，颜色的随机扰动或者输入的非线性几何变形。

### 语音识别
由最开始的‘隐马尔可夫模型’和‘高斯混合模型’组成的 HMM-GMM结合方法。最终发展到了end-to-end的深度卷积神经网络方法，其中利用了卷积的方法，把输入的频谱当成一个向量，其中一个轴对应着时间，另外一个轴对应着谱分量的频率。

### NlP 自然语言处理
- n-gram：一个n-gram是一个包含n个标记的序列。但他特别容易陷入维数灾难。在one-hot向量空间中的距离彼此相同。
- 神经语音模型使用短列表来减轻计算的高成本，但仅限于常用的词。
- 通过建立词的类别树，是计算代价由v降到log(v)。使用分层softmax带来了计算上的好处。
- 重要采样可以加速大稀疏输出层的训练，其中输出是稀疏向量，而不是n选1的向量。
- 结合n-gram和神经语言模型。n-gram具有更高的模型容量，同时处理样本的计算量很小。
- 神经机器翻译中的 编码-解码模式 和 注意力attention模型可以实现成功的语音翻译。

## 第3部分 第13章 深度学习研究
### 线性因子模型
- 概率PCA和因子分析是上述等式的特殊情况。
- 独立成分分析 Independent component analysis ICA，是最古老的表示学习方法之一。
潜在因子的先验P(h)，必须由用户提供。
独立子空间分析：地质ICA
- 慢特征分析slow feature analysis是使用来自时间信号的信息学习不变特征的线性因子模型。
与场景中描述作用的某个单独量度相比，场景的重要特征通常变化的非常缓慢。
在运行SFA之前通常使用非线性的基来扩充学习非线性特征。
主要的优点是，在深度非线性条件下，仍然可以在理论上预测SFA能够学习那些特征。
1. 学习特征具有零均值约束
2. 特征具有单方差约束
3. 学习的特征彼此去相关
- 稀疏编码：稀疏编码是指在模型中推断h值的过程

线性因子模型是最简单的生成模型和学习数据表示的最简单模型。这些线性因子模型可以扩充到自编码器网络和深度概率模型，它们可以执行相同任务但具有更强大和更灵活的模型组

---

## Insight of a Six Layered Neural Network along with other AI Techniques for Path Planning Strategy of a Robot

> Das S S, Parhi D R, Mohanty S. Insight of a Six Layered Neural Network along with other AI Techniques for Path Planning Strategy of a Robot[J]. Emerging Trends in Engineering, Science and Manufacturing (ETESM-2018), IGIT Sarang, India, 2018.

### six layered neural network method for robot navigation

- input layer: 4 neurons, including distance measurement in front, left and right. The last input neuron measures the target angle.
- the first and second intermediate layers have 18 neurons. the third layer has seven neurons. the fourth has three neurons. the output layer has a single neuron to steer the direction of locomotion of the robot.
- activation function is SIGMOID FUNCTION, which given by 
```math
f\left( x \right) = \frac{1}{{1 + {e^{ - x}}}}.
```
- Then, a full-connectity network was to predict the angle.During the training of the neural network, nearly one thousand training patterns are used to cope up with the unknown scenario during path planning.

## end

---
# 10.16

## # neural-fuzzy control

将神经网络与模糊控制逻辑相结合。神经网络具有自适应和自学习的特点，但是它在表示规则方面不够强大。模糊控制具有更强的抗干扰性和鲁棒性，但是传统模糊控制逻辑的维数很低。
因此，Fuzzy-Neural control method 融合二者来克服其分别的缺点。


---
# 10.17
### to-do list
- [x] - simulation for paper
- [x] - code for leetcode
- [x] - bilibili
- [x] - 算法导论一章
### 模糊控制
#### 优点
- 模糊控制是一种基于规则的控制，它直接采用语言型控制规则，出发点是现场操作人员的控制经验或相关专家的知识，在设计中不需要建立被控对象的精确的数学模型，因而使得控制机理和策略易于接受与理解，设计简单，便于应用。
- 由工业过程的定性认识出发，比较容易建立语言控制规则，因而模糊控制对那些数学模型难以获取，动态特性不易掌握或变化非常显著的对象非常适用。
- 有较强的鲁棒性和抗干扰性

#### 缺点
- 由于出发点是现场操作人员的控制经验或者相关专家的知识，无法利用大数据的优势而完成自学习。

### paper
在这部分，我们将展示我们的仿真实验结果，同时对结果进行分析比较。
仿真实验分为两个部分：离线训练和在线规划。
首先，通过bp算法，我们利用keras对MLP进行训练。可以看出对mlp的拟合实际上是神经网络的回归问题。因此，训练过程中的mean squared error 被选为loss function。同时 mse也被选为metric函数，which用来评估当前训练模型的性能。评价函数和损失函数类似，只不过评价函数的结果不会应用到训练过程中。Adam被选为optimizer，which is 一种可以计算每个参数的自适应学习率的方法，在最近几年被大量使用。

在在线规划器部分，为了验证所提出的算法，两个仿真分别从运行时间和规避效果两个方面评估实时性和有效性。我们为固定翼无人机构建了2d的飞行环境with 障碍物，同时一条包含n个航迹点的任务轨迹被预先置入固定翼无人机。



### LeetCode 最大自序和

思想：求连续序列的最大值，从前向后加，遇到<0的值则放弃该子串。用res存储遇到过的最大的子串。
good idea

---

# 10.18
### To-Do list
- [x] 连续路线的障碍物规避仿真图
- [ ] 运行时间对比（加速问题求解时间）

### 堆排序
堆排序构建完全二叉树（大顶堆）。从数组的末尾逐一向前遍历，比较孩子节点和根节点的大小。
如果根节点小于孩子节点，则swap两个节点的值，并检查新的左子树是不是大顶堆，如果不是则将新的左子树重新建堆。不断进行递归，建树构建大顶堆。

构建完成后将根节点，即最大的值，与数组末尾的值交换。然后，继续构建大顶推，如此循环往复。

时间复杂度：o（nlogn)。空间复杂度o（n）

# 10.20
nothing
# 10.21
### To-Do list
- [x] 连续路线的障碍物规避仿真图
- [x] 减少神经元个数
15个最终可以实现比较好的性能
- [x] 航模无人机平台重新设计：nuc+intel-realsense相机+3D雷达+毫米波雷达。
- [x] 因诺：毫米波雷达数据协议解析（刘韩+张景秦），原来的相机启动及保存（陆磊），连接wifi控制tx2和nuc（Dck）。ads-b暂时放一放。
- [ ] 陆磊 3d雷达 realsense相机
# 10.22
- 共轭迭代：在共轭迭代中，我们寻求一个和先前下降方向共轭的搜索方向。非线性共轭迭代，包括一些偶尔的重设，其被证明对训练神经网络是有效的。
### 优化策略和元算法
1. 批标准化：批标准化是训练深度神经网络中非常重要的创新，其提出了一种重参数化的方法。重参数化显著减少了多层之间协调更新的问题。通过批标准化，将高斯变成单位高斯，消除了均值和方差的影响。但是针对这一批单元激活函数，我们为其赋予了同样的增益权重和偏移量，使其具有任意的均值和方差。批标准化降低了神经元的表达能力。
2. 坐标下降：交替固定某个（些）坐标，求取整个代价函数最小值的方法。保证可以达到（局部）最小值。
3. Polyak平均：会平均优化算法在参数空间访问轨迹中的几个点。
4. 监督预训练：在正式训练之前，进行参数网络的预训练。训练一个简单的模型然后使模型更加复杂也许会更有效。贪心算法监督预训练将整个问题分解成几个部分，将几个部分训练至最优。再将几个部分结合起来。然而，这种方法并不能保证是最优，但这种方法是一种很好的参数预训练方法极大地加速了算法的训练。预训练对参数的泛化能力是有帮助的。教师-学生的迁移学习方法由这个思路扩展而来。其通过复杂的教师网络来逐层监督学生的神经网络。这样可以大大减小网络的复杂，去除冗余的参数。
5. 一个有助于优化的模型，往往比一个优秀的算法更加重要。
6. 延拓法：通过预先训练一个较容易的模型，再逐渐增加目标函数的复杂度，来逐渐接近最根本（复杂）的优化函数。即将一个非凸问题转化成一个模糊的凸问题再逐渐逼近非凸问题。但是这样的缺点是，np-hard问题永远是np-hard问题。仍然有可能陷入局部最小值。
### 序列建模：循环和迭代神经网络
- 循环神经网络是专门用于处理序列信息的神经网络。通过计算图的形式来描述循环神经网络的计算过程。本质上，任何涉及循环的函数都可以被视作循环神经网络。
- 学成的模型始终具有相同的输入大小，指定的是从一种到另外一种状态的转移。不是在可变长度的输入下进行操作。
- 可以在每个时间步，使用相同参数的相同转移函数。
- RNN中的反向传播算法是‘通过时间反向传播’BPTT。
- 导师驱动过程和输出循环网络：使用导师驱动过程和自由运行的输入进行训练。可以提升训练效果。或者使用课程学习的思想。逐步使用更多生成值。

### 基于上下文的rnn序列建模
RNN的输入可能是一个固定的X，也可能是一个序列，

```math
\{x^0,x^1, ... , x^t\}。
```
但可变长度的x序列只能生成对应长度的y（输出）序列。
# 10.23

```math
min_{x} \ \frac{1}{2}| Ax-b|^2 + \mu |x|_1

A \in \mathbb{R}^{m \times n}

b \in \mathbb{R}^{n}
```

非凸优化问题

```math
f\left( {{x_e},{y_e}} \right) =  \sum\limits_{i = 1}^{15} {\left( {\frac{{w_4^i}}{{w_1^i*{e^{w_1^i{x_e} + w_2^i{y_e}}} + 1}}} \right)} +b
```
求解f（xe，ye），使
```math
f(x_e,y_e)
```
最大时，（xe，ye）的解。
# 10.24
### 拟牛顿法
牛顿法中的Hesse矩阵``` h```在稠密时求逆计算量大，也有可能没有逆（Hesse矩阵非正定）。拟牛顿法提出，用不含二阶导数的矩阵 
```math
U_t
```
 替代牛顿法中的 
```math
H^{−1}_t
```
，然后沿搜索方向 
```math
−U_t g_t
```
 做一维搜索。根据不同的 
```math
U_t
```
 构造方法有不同的拟牛顿法。
### 交叉熵
1. 信息论：越不可能的事件发生了，我们获取到的信息量就越大。越可能发生的事件发生了，我们获取到的信息量就越小。信息量的定义：

```math
L(x_0) = log(p(x_0))
```
熵就是所有信息量的期望。
2. 相对熵又被称为KL散度，我们认为两个分布越接近，两个分布的相对熵就越小。将相对熵分解成两部分，第一部分是P的熵，而第二部分是交叉熵。我们利用交叉熵来评估模型预测结果与真实结果的差异，即loss function。

### 双向RNN
双向循环神经网络，不仅仅有前向传播的h隐藏层，还有反向传播的g层。这使得输出单元o^t可以利用过去的信息和未来的信息。这样可以捕捉到大多局部的信息并同时依赖过去的新。

### 基于编码-解码的序列到序列架构
利用编码器将输入序列编码为一个固定长度的序列C，再将C解码（解释）为输出。这样输入和输出可以有不同的长度。
这种架构一个明显的不足是：上下文序列C可能会由于维度过短而不能良好的表示一个长序列。

后来又加入了注意力机制将序列c中的元素与输出序列的元素相关联。
### 深度循环网络
深度循环网络可以看成是将MLP加入到RNN的架构中。一般来说RNN可以被分成三个部分：
- 从输入到隐藏状态
- 从前一隐藏状态到下一隐藏状态
- 从隐藏状态到输出

将某一子模块用MLP替换可以得到更深的DEEP RNN。理论上这种可以使RNN的表示能力更强，但伴随的是更加复杂的训练难度。

### 递归神经网络
递归神经网络将架构设计成为树形，这样架构使深度由o(n)变成了o(logn)。
### 长期依赖的挑战
RNN的函数可以使用非线性的。但是多次非线性的函数累乘可能使得重复组合函数变得高度非线性化（极端非线性化行为）。

保留长期的信息与短期信息的波动是相悖的。短期内的波动可能会覆盖掉长期依赖的小幅值信息。
我们可以通过时间维度的跳跃连接、渗漏单元来保持长远信息的影响。
- 时间维度的跳跃连接：
即将长久的隐藏单元连接到当前隐藏单元
- 渗漏单元：利用滑动平均来保留之前信息的影响，通过一个参数μ来控制更新信息和保留信息比例。

### 长短期记忆和门控RNN（Gated RNN）
- LSTM中加入了更多的参数和控制信息流动的门控单元系统。细胞之间彼此连接，代替循环网络中的循环隐藏层。门控单元一般设置成为sigmoid函数。通过sigmoid函数输出的值[0,1]来控制信息更新，或是是否要放弃这次更新。输入门，遗忘门和输出门协调控制、参与网络的更新。
### 截断梯度
截断梯度以避免梯度爆炸或者梯度消失的情况。通过尽量将梯度的导数比控制在1左右以保证梯度不会过大或者过小。
### 外显记忆
神经网络擅长表示输入和输出时间隐藏的关系。但是它并不擅长表示显式的信息，例如通过记忆和几个歌词回忆起是哪首歌。
在神经网络添加一个独立的记忆网络（任务网络），任务网络可以选择读取或写入的特定内存地址，似乎可以使信息在更长的时间内流动。
# 10.28
A LSTM Neural Network applied to Mobile Robots
Path Planning
> Nicola, Fiorato, Yasutaka Fujimoto, and Roberto Oboe. "A LSTM Neural Network applied to Mobile Robots Path Planning." 2018 IEEE 16th International Conference on Industrial Informatics (INDIN). IEEE, 2018.

- **global path planner:** we want to realize an online search agent based on the
Long Short-Term Memory neural network with a supervised
learning approach where the ground truth is provided by the
A* algorithm. Datasets are realized with the path solutions found by the supervisor, then the neural network is trained and
eventually it is deployed as an online agent.
- **conceptual steps on how to consider path planning:**

![image](DF10EAA787B14012835D97E4DFF34B94)输入是：current state，environment info， goal point info组成的序列。
输出：direction selected的决策序列。

LSTM cell 的表示：
![image](969F502C39054351BB649111B7CD08E6)
**The deep LSTM neural networks considered**
![image](7254CD9E19874C0E9BF9C075463D5D19)

> **递归**：卷积神经网络(CNN)中全连接层(FC layer)的作用：卷积层(Convolutional layer)主要是用一个采样器从输入数据中采集关键数据内容；
池化层(Pooling layer)则是对卷积层结果的压缩得到更加重要的特征，同时还能有效控制过拟合。

文章对比了LSTM和MLP的train loss， train accuracy， validation accuracy。
以及两者的权重对比。相比A* algorithm，基于LSTM的方法相比来说成功率较低。
![image](11C6C0F20D054951AE4572EAB80FAB1B)

### 广度优先搜索 BFS
1. 建立一个队列，将开始点加入到队列中。
2. u = 队头，然后队头出队，依次将未访问过的，相邻的节点加入队列。
3. 重复step2，直到寻找到最终的终点。

# 10.29
lstm-mdl模型：The LSTM-MDL model [9] is a recurrent neural network
(using LSTM cells1
) that is used to parameterize a mixture density output layer (MDL)
### 循环神经网络RNN::微软人工智能公开课
- 前馈式神经网络feedforward neural network: 神经网络==复合函数，计算图的用意：方便反向传播梯度。
![image](7F90B3D581F34C5F8755EA4CCC6FDBFA)
- 循环神经网络（Rnn，recurrent neural network）：最早出现于1982年，hopfield network（极端的神经网络）。
循环神经网络≠递归神经网络
循环神经网络的参数是共享的，各个时刻的参数是一致的。循环神经网络是可以展开的。![image](7FC932182D7A405289D4168D72615064)
循环神经网络有两个输入，激活函数是双曲正切函数tanh。
- rnn语言模型：
语言模型language model 估计自然语言句子的概率。
困惑度：交叉熵= P(x)log(q(x))
BPTT算法back propagation through time：1.构建loss function，然后反向求梯度。
门控计算单元（GRU）：与lstm相比gru只有一个信息单元。
RRN的变式：最小门控单元（MINIMAL Gated UNit），简单循环单元（simple recurrent unit）。
- 层叠RNN网络：
![image](B43D4120634946088853DA14B2B78930)增加RNN的层数导致训练慢，同时容易过拟合。
- RNN应用示例：SRU效果似乎更好，强于初始的LSTM。多层网络使效果更好，但是训练速度大大增加。
使用RNN可以自动生成对联。

### 卷积神经网络
- 卷积神经网络中的一层包括，卷积层，激活函数，池化。
- 训练网络的方式取决于loss function（任务：分类或者是回归）。

# 10.30
无人机公司出差
# 10.31
无人机公司出差
# 11.1
*新的一个月开始了*
![image](8027CA39A7BE4868B5E195963BA9985D)
**JinwenHU:**
对于每个指标函数，用加强学习和你的方法是等价的

**JinwenHU:**
但是用你的方法的好处是，用同一批样本可分别学习多个指标模型，进而随意组合，权重改变后，不用再学习，直接求解。加强学习不行，必须重新设计收益函数，重新学习

### 非线性优化问题
20世纪80年代以来，随着计算机技术的快速发展，非线性规划方法取得了长足进步，在信赖域法、稀疏拟牛顿法、并行计算、内点法和有限存储法等领域取得了丰硕的成果。

```math
argmax \ \varphi(p_e^*) = \eta_1 * F(p_e^*) + \eta_2 * dirvation + \eta_3 *\varepsilon(p_e^*) \\ 
\ \\
s.t. F(p_e^*,x_s^i, y_s^i, p_e^*)>R
```
# 11.3 周日
> 赵老师
你的文章的立意还是不够准确，你必须把自己整个系统的难点提炼成你要解决的问题，这样文章才立意准确。难点==待解决问题。

# 11.4 周一
因诺飞无人机

# 11.5 周二
教研室搬家

# 11.6 周三
> Yamashita, Takahiro, and Takeshi Nishida. "Path Planning using Multilayer Neural Network and Rapidly-exploring Random Tree."

- 利用RRT算法生成路径。Subsequently, an RRT was applied to these sets to generate
1100
- 算法流程：给定始发点和终点，生成两点之间的终点。循环两次。生成5个路径点。
- We proposed a path-planning method that combines
a metaheuristic method and an NN in an environment
where obstacles exist. In this method, a path is generated
by the L-MLN, which learns the path generated by
an RRT. The proposed method avoids the limitation of
metaheuristics i.e., the low reproducibility of path generation
and the limitation of NN i.e., the amount of training
data sets, and determines a quasi-optimal solution at high
speed and high quality.

> Motion Planning of Autonomous Mobile Robot Using Recurrent
Fuzzy Neural Network Trained by Extended Kalman Filter

 A planner based on the recurrent fuzzy neural network (RFNN) is
designed to program trajectory and motion of mobile robots to reach target。

A real-time program strategy in unknown dynamic surrounding is proposed, i.e., without any previous offline computation.

recurrent fuzzy neural network （RFNN）structure is designed.
![image](1CB2D254207A4A838778EA965AA4A13C)
文章中使用的RFNN包括5层：第一层，包含四个神经元的输入层：![image](09D6156AD6904A53A397F3A4972758F4)

第二层：membership layer，标准化输入层。
dg (do) can be divided into far and near depending on the distance between mobile robot and goal (the nearest obstacle). θ_g (θ_o) can be divided into left and right depending on goal’s position (the nearest obstacle’s position) related to the mobile robot’s front direction.

第三层（模糊逻辑层、Fuzzy Rule Layer）：局部的有延时的内反馈来形成一个循环结构，使用一个参数来控制更新比例。

第四层（Consequent Layer）：本层将consequent node进行线性组合。

第五层：输出层含有两个节点，分别表示两个履带的线速度。

> Lu, Kaida, Yan Zhang, and Huiyuan Li. "Research Status and Development Trend of Path Planning Algorithm for Unmanned Vehicles." Journal of Physics: Conference Series. Vol. 1213. No. 3. IOP Publishing, 2019.

# 11.7 周四
准备mj-ppt

# 11.8 周五
#### minji会议
北航：改进的ssd+kcf以多线程的结构并行处理，加快了算法跟踪的帧率。
![image](1CBAECCB7DD341D9991365CF307FE5C8)
- 对潜在的碰撞区域进行分类。
- 威胁评估并引入时间规避决策阈值。
![image](2A9647A56C7740559E73AB0C688AC6CF)
一整套障碍物规避算法

- 对照合同具体的完成情况（理论，技术及平台）按照合同的专题对比完成情况。
- 项目总体进展的一个报告


# 11.9 周六

# 11.11 周一
Dubins model：
![image](FBF7FEBA44264CD6858C8CC99C9B6FFD)
- *通过图像（雷达）序列lstm来预测飞机的动作，作为感知规避的辅助应用。*

##### 多障碍物的情况

![image](3404E047595C4F1780BB840A31F96204)
目前算法的缺陷：
1. 样本是单障碍物的情形。面对多障碍物的情况比较难处理，难以聚类。如果将样本改成多障碍物的网络会变得复杂（输入序列的长度是可变的，但是可以使用lstm？2.如果将多个障碍物变成<角度，距离>信息，相当于损失了一部分信息。）。
2. 神经网络难以求极值。
3. 如果使用非线性优化，那么运行速度比较慢。
![image](304E9D73459942D5BB77042B8A02A074)

```math
Distance = F(p_s,p_e,p_o) \\
F(p_s,p_e,p_o)是拟合出来的神经网络\\
判断两个障碍物p_{oi},p_{oj},是否存在p_e使\\
Distance_i = F(p_s,p_e,p_{oi})>R_{min}\\
Distance_j = F(p_s,p_e,p_{oj})>R_{min}
```
非线性规划方法有信赖域法、稀疏拟牛顿法、并行计算、内点法和有限存储法等。
- 这几种方法都是通过**数值或者迭代**的方法。
- 内点法中有一个惩罚函数，用于描述凸集。与单纯形法不同，它通过遍历内部可行区域来搜索最优解
![image](WEBRESOURCE13931c8fc70eb4a6d821da12cb68f48f)
改进：
1. 用多障碍物情形生成样本。然后将环境转化成方向，然后控制无人机的未来航点。（如果将多个障碍物变成<角度，距离>信息，相当于损失了一部分信息。）
2. 输入序列的长度是可变的，但是可以使用lstm？

# 11.12 周二

##### 报销
200+197+168+172.8+199+22.8

出租车：69.4+75.1+104.4+83.8+110.90+94.3+191.60（李卓一）+85.6+75.1+78.4+77.5+68.8+70.9+69.4+73.6+78.4+96.8（dck）+76+78.70
##### 隐马尔可夫模型HMM
三大假设：1. 齐次马尔科夫假设。2.观测独立性假设。3.参数不变性假设。
隐马尔可夫模型主要有三大要素，分别是：**初始状态向量π，状态转移概率矩阵A和观测概率矩阵B**。此三大要素决定了一个模型。
使用HMM模型时我们的问题一般有这两个特征：

１）我们的问题是基于序列的，比如时间序列，或者状态序列。

２）我们的问题中有两类数据，一类序列数据是可以观测到的，即观测序列；而另一类数据是不能观察到的，即隐藏状态序列，简称状态序列。

- 根据B，由隐藏序列得到可观测的状态序列。
- 状态转移概率矩阵A描述了不同状态之间的转移概率。
- π是HMM的初始状态向量。

##### sequence to sequence实现
与Seq2Seq框架相对的还有一个CTC，CTC主要是利用序列局部的信息，查找与序列相对的另外一个具有一对一对应关系（强相关，具有唯一性）的序列，比较适用于语音识别、OCR等场景。

attention模型最大的不同在于Encoder将输入编码成一个向量的序列，而在解码的时候，每一步都会选择性的从向量序列中挑选一个子集进行输出预测，这样，在产生每一个输出的时候，都能找到当前输入对应的应该重点关注的序列信息，也就是说，每一个输出单词在计算的时候，参考的语义编码向量c都是不一样的，所以说它们的注意力焦点是不一样的。

##### 毕设开题
- 第一章绪论（国内外研究现状）
- 第二章lstm障碍物轨迹预测与安全包络建模
- 第三章fw nn路径规划算法
- 第四章仿真与实验

### 11.13 周三
> SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction

> LSTM-based Deep Learning Model for Civil Aircraft Position and
Attitude Prediction Approach

> Bency, Mayur J., Ahmed H. Qureshi, and Michael C. Yip. "Neural Path Planning: Fixed Time, Near-Optimal Path Generation via Oracle Imitation." arXiv preprint arXiv:1904.11102 (2019).

- 问题描述

在文章中，使用configuration space来构建environment。相比笛卡尔空间，C space对于高维度的运动规划更有优势。所以对于障碍物的检测，我们也将碰撞检测映射到c space中。
对于environment，我们假设障碍物的信息是已知的。以下是配置空间及空间轨迹的定义：![image](54A65BD5BEA0487685D7EFDAF8D91765)

- 提出的算法

使用了stacked Long Short Term Memory (LSTM) layers 和一个连接output的全连接层构成了神经网络

- 创建训练数据&离线学习

文章利用A*算法生成最有轨迹作为数据集，即Oracle。对配置空间进行采样来获得样本点集，然后在样本点集中随机选择初始位置和终止位置来生成路径样本。
利用生成好的路径样本训练LSTM网络。
- 双向轨迹生成方法

在online轨迹生成的过程中，文章使用了双向生成的方法，即从起始点与终点同时生成无碰撞的轨迹，直至两个轨迹相交。最终生成一条完整的无碰撞轨迹。

- 轨迹优化

1.移除轨迹中与障碍物碰撞的点。2.平滑轨迹：即将轨迹尽量连接成直线以减少路径总长度。

- 结果

仿真环境设置为100*100的2D网格空间，并添加了凸/凹的障碍物。文章使用RRT,A星与提出的算法进行对比。

### 11.14 周四

> Kelchtermans, Klaas, and Tinne Tuytelaars. "How hard is it to cross the room?--Training (Recurrent) Neural Networks to steer a UAV." arXiv preprint arXiv:1702.07600 (2017).

- input from a forward looking camera.

### 11.15 周五
一般来说，障碍物间的距离被用来作为聚类的度量标准。然而，在本篇文章中，由于飞机的飞行路线已经提前给定，障碍物与飞行路线的最短距离可以被计算出来，这也是影响飞行安全的重要因素。因此，与大多数距离方法不同的是，我们使用两种计算距离的方法，即欧氏距离和障碍物与路径的最短距离，作为聚类的度量标准。
1. 如果障碍物的间距小于二倍最小规避距离，这意味着无人机从障碍物之间穿过的情况是不符合安全要求的。因此，相邻的障碍物将被聚类。
2. 如果飞行路径的左右两侧均存在障碍物，且两侧的障碍物与飞行路径的距离均小于最小规避距离，则这两个障碍物（可能相距很远）将被聚类。

将多个障碍物聚集成一簇后，一个更大的障碍物安全包络将会被生成。

### 11.16 周六
- 安全包络如何设置？
- 轨迹碰撞区域：

### 11.17-11.20 沈阳出差

### 11.21 周四

In this paper, OEM is designed for the obstacle geometrical modelling based on several brief principles. Generally, there are two kinds of enveloping modes using a unified primitive in OEM: the single enveloping and the team enveloping. The former is designed for the simple, isolated obstacles, whereas the latter for the complicated, clustered obstacles. The general idea of OEM is to try the former first by checking if it obeys the principles below or not; if not, the latter is applied. If the latter fails, then there are no other options. Under this situation, the spanning line SG has to be revised in the higher level to avoid the local irrationality. This is beyond the discussion of this paper. A suggestion is that when creating a preliminary path consisting of a series of points by the heuristic algorithm in the high level, try the concept of entropy H defined below as one of the optimisation criteria, a supplement to the common factors including the path length, smoothness, collision possibility, danger degree, etc.

> Moon, Jongki, and J. V. R. Prasad. "Minimum-time approach to obstacle avoidance constrained by envelope protection for autonomous UAVs." Mechatronics 21.5 (2011): 861-875.

### 11.22 周五

### 11.24 周日
**Sample Efficient Learning of Path Following and Obstacle Avoidance Behavior for Quadrotors**
> Stevšić, Stefan, et al. "Sample efficient learning of path following and obstacle avoidance behavior for quadrotors." IEEE Robotics and Automation Letters 3.4 (2018): 3852-3859.

- robot model：![image](C30F921EFD1D46C9B9F31A5A1471E562)
- assume that the system has sensors, such as a laser range finder. Therefore, an observtion vector can be obtained,
```math
O_t=[d_t,v_t,l_t], l_t\in R^{40}.
```

We have proposed a method for learning control policies
using neural networks in imitation learning settings. The approach leverages a time-free MPCC path following controller
as a supervisor in both off-policy and on-policy learning. We
experimentally verified that the approach converges to stable
policies which can be rolled out successfully to unseen environments both in simulation and in the real-world. Furthermore, we
demonstrated that the policies generalize well to unseen environments and have initially explored the possibility to roll out
policies in dynamic environments.

**An improved recurrent neural network for unmanned underwater vehicle online obstacle avoidance**
> Lin, Changjian, et al. "An improved recurrent neural network for unmanned underwater vehicle online obstacle avoidance." Ocean Engineering 189 (2019): 106327.

- However, traditional RNNs
cannot extract environment features effectively, and contain many
network parameters. To overcome these problems, this paper proposes
an RNN with convolution (CRNN) in which a convolution connection
replaces the full connection between adjacent layers of the RNN

### 11.25 周一
> Zheng W, Wang H B, Zhang Z M, et al. Multi-layer Feed-forward Neural Network Deep Learning Control with Hybrid Position and Virtual-force Algorithm for Mobile Robot Obstacle Avoidance[J]. International Journal of Control, Automation and Systems, 2019, 17(4): 1007-1018.

### 11.26 周二

![image](94F52FACF69744E4BEE1F4D4168961D7)

In order to reduce the number of plans and the distance of detours as much as possible, this paper designs a dynamic estimation method of obstacle circle, as follows:

(1)
Considering that obstacles are mostly continuous spatial geometries, a relatively small radius R1 is first used as the radius of the obstruction circle, and the obstruction circle is used for Dubins path planning.

 
(2)
Starting from the previous route planning, if there is an obstacle within the safe distance ahead of UAV Flight T, the obstacle is the same obstacle as the last estimated obstacle. At this time, combining the positions of the obstacles detected twice, the Dubins path planning is performed again using the radius R2 as a new obstacle circle (R2 > R1);

 
(3)
If no obstacle was detected during UAV flight T from the previous route planning, clear the historical information, and deal with new obstacles when obstacles are re-detected, as in step 1).

A clothoid or cornu spiral is a curve whose curvature changes
linearly with its length (Figure 1). A parameter representation
is:![image](0F2534FA92954BEFBF6627CBEEC3EA7C)
with：
![image](7A41763A61B64CE5AAB8583153BE65EF)
where a represents a scale factor which defines the rate of
change of the curvature and hence the size of the clothoid. It is
well known and straightforward to verify that the arc length and
the curvature of the curve are:
![image](6217D475B9CB49AF8F366F753EC1BBB8)

### 11.27 周三
> Kikutis R, Stankūnas J, Rudinskas D, Masiulionis T. Adaptation of Dubins Paths for UA V Ground Obstacle Avoidance When Using a Low 
Cost On-Board GNSS Sensor. Sensors 2017; 17(10): 23 p., https://doi.org/10.3390/s17102223

一种将由离散的点集组成的路径转化成Dubins曲线路径的算法。
- 考虑到实际工程中，车轮的转向是一个连续的过程，不可能从0直接变成最大转向角。所以坐着提出了一个评价约束，以保证生成可以行驶的路径。最终算法证明了实验的可行性。
