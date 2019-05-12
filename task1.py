
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


my_arr = np.arange(1000000)


# In[3]:


my_list = list(range(1000000))


# In[4]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_arr2 = my_arr * 2')


# In[5]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_list2 = [x * 2 for x in my_list]')


# In[6]:


# Numpy 的  ndarry: 一种多维数组对象


# In[7]:


# 生成随机数据的数组
data = np.random.randn(2, 3)


# In[8]:


data


# In[9]:


data * 10


# In[10]:


data + data


# In[11]:


# ndarray 是一个通用的同构数据多维器，其中所有元素必须是相同类型
data.shape


# In[12]:


# (2, 3) 二维数组，每一维有3个元素
# 查看数据类型
data.dtype


# In[13]:


# 创建一个列表，使用 np.array() 函数转换为 numpy 数组
data1 = [6, 7.5, 8, 0, 1]

arr1 = np.array(data1)

arr1


# In[14]:


# arr1 中的元素全部转换了成了浮点数元素


# In[15]:


# 嵌套序列，如有一组等长的列表组成的列表，将转换成一个多维数组
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]


# In[16]:


arr2 = np.array(data2)


# In[17]:


arr2


# In[18]:


# arr2 二维数组，每一维有4个元素，这就原始的列表一致，两个等长列表
arr2.ndim


# In[19]:


arr2.shape


# In[20]:


arr1.dtype


# In[21]:


arr2.dtype


# In[22]:


# 使用 np.zeros() 和 np.zones() 可以创建指定长度或形状的数组，分别为全0或全1
#  np.empty() 可以创建一个没有任何具体值的数组
np.zeros(10)


# In[23]:


np.zeros((3, 6))


# In[24]:


np.empty((2, 3, 2))


# In[25]:


# empyt() 大多会返回一些未初始化的随机值
np.arange(15)


# In[26]:


# dtype 数据类型是一种特殊对象，ndarrary 将一块内存解释为特定数据类型所需的信息

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)

print(arr1.dtype)
print(arr2.dtype)


# In[27]:


# 可以通过 数组的 astype() 方法进行转换成其他  dtype
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)

float_arr = arr.astype(np.float64)
print(float_arr.dtype)


# In[29]:


# 整数被转换成浮点数，如果浮点数转换成整数，小数点后将被截取删除
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
print(arr)
print(arr.astype(np.int32))


# In[30]:


# 字符串里如果全是数字，也能用 astype 转换为数值
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# In[31]:


# 上面所写的 float 在 numpy 里会转换成  np.float64

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)

print(int_array)
print(int_array.astype(calibers.dtype))


# In[33]:


empty_uint32 = np.empty(8, dtype='u4')
empty_uint32


# # Numpy数组的运算

# In[37]:


arr = np.array([[1., 2., 3.], [4., 5., 6.]])


# In[38]:


arr


# In[39]:


arr * arr


# In[40]:


arr - arr


# In[41]:


# 数组与标量的算术去处会将标量值传播到各个元素
1 / arr


# In[42]:


arr ** 0.5


# In[44]:


# 大小相同的数组之间的比较会生成布尔值数组
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])


# In[45]:


arr2


# In[46]:


arr2 > arr


# In[47]:


# 不同大小的数组之间的去处称为广播


# ## 基本索引和切片

# In[48]:


arr = np.arange(10)


# In[49]:


arr


# In[50]:


arr[5]


# In[51]:


arr[5:8]


# In[52]:


arr[5:8] = 12


# In[53]:


arr


# In[54]:


# 将一个标量赋值给一个切片时，该值会自动传播到整个切片选区
# 与列表不同的是，数组切片是原始数组的视图，这意味着数据
# 不会被复制，视图上的任何修改都会直接反映到源数组上
arr_slice = arr[5:8]


# In[55]:


# 当修改 arr_slice 变动会在原始数组 arr 中体现
arr_slice[1] = 123345
# arr_slice[1] 是原始数组的 6
arr


# In[56]:


# 切片[:] 表示整个切片范围
arr_slice[:] = 64
# 即是将原来的 12 12345 12 变为 64
arr


# In[57]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]


# In[58]:


"""
高维数组时，要注意索引表示的范围
单一维度时，索引表示的是标量，高维时，第一个索引表示的一个数组

索引单个数组时，可使用 arr2d[0][2] == arr2d[0, 2]
使用逗号（，）隔开索引也可以，两者是等价的
"""
arr2d[0][2]


# In[59]:


arr2d[0, 2]


# In[70]:


# 如上所说，多维数组如果省略了后面的索引
# 返回的是低一维的 ndarray ， 其包含高一级维度的所有数据
arr3d = np.array([[[1, 2, 3,], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


# In[71]:


arr3d


# In[72]:


arr3d[0]


# In[73]:


old_values = arr3d[0].copy()


# In[74]:


arr3d[0] = 42


# In[75]:


arr3d


# In[76]:


arr3d[0] = old_values


# In[77]:


arr3d


# In[78]:


arr3d[1, 0]


# In[79]:


x = arr3d[1]


# In[80]:


x


# In[81]:


x[0]


# ## 切片索引

# In[82]:


arr


# In[83]:


arr[1:6]


# In[84]:


arr2d


# In[85]:


arr2d[:2]


# In[86]:


arr2d[1, :2]


# In[87]:


"""
对多维数组进行操作与索引引值类似，单一索引将选取最底维度的数据
以逗号分隔可以更精准取得数据

对切片表达式的赋值操作也会被扩散到整个选区
"""
arr2d[:2, 1:]=0
arr2d


# ## 布尔值索引

# In[88]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[89]:


data = np.random.randn(7, 4)


# In[90]:


names


# In[91]:


data


# In[92]:


names == 'Bob'


# In[93]:


# 布尔值可用来作为数组索引
data[names == 'Bob']


# In[94]:


data[names == 'Bob', 2:]


# In[95]:


data[names == 'Bob', 3]


# In[96]:


# 上面逗号隔开的是索引，取切片，或者单独取元素
names != 'Bob'


# In[99]:


data[~(names == 'Bob')]


# In[100]:


cond = names == 'Bob'
print(cond)


# In[101]:


data[~cond]


# In[102]:


mask = (names == 'Bob') | (names == 'Will')


# In[103]:


mask


# In[104]:


data[mask]


# In[105]:


names == 'Bob'


# In[106]:


names == 'Will'


# In[107]:


# 两者合一即将为真的位取真


# In[108]:


data[data < 0] = 0


# In[109]:


data


# In[110]:


data[names != 'Joe'] = 7
data


# ## 花式索引

# In[111]:


arr = np.empty((8, 4))


# In[112]:


arr


# In[113]:


for i in range(8):
    arr[i] = i


# In[114]:


arr


# In[115]:


arr[[4, 3, 0, 6]]


# In[116]:


arr[[-3, -5, -7]]


# In[117]:


arr = np.arange(32).reshape((8, 4))


# In[118]:


arr


# In[119]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[120]:


# (1, 0), (5, 3), (7, 1), (2, 2)
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# In[121]:


"""
解释上一个索引
    [[1, 5, 7, 2]]
    选取1， 5, 7， 2索引的数组
    [:, [0, 3, 1, 2]]
    每个索引的数据都做切片
    按[0, 3, 1, 2]的索引重新排列切片元素
"""


# ## 数组转置和轴对换

# In[122]:


arr = np.arange(15).reshape((3, 5))


# In[123]:


arr


# In[124]:


arr.T 


# In[125]:


arr = np.random.randn(6, 3)


# In[126]:


arr


# In[127]:


np.dot(arr.T, arr)


# In[128]:


arr = np.arange(16).reshape((2, 2, 4))


# In[129]:


arr


# In[130]:


arr.transpose((1, 0, 2))


# In[131]:


# 简单转换可以用 T，其进行的是轴对换
arr


# In[132]:


arr.swapaxes(1, 2)


# ## 通用函数：快速的元素级数组函数

# In[133]:


arr = np.arange(10)
arr


# In[134]:


np.sqrt(arr)


# In[135]:


np.exp(arr)


# In[136]:


# 二元函数
x = np.random.randn(8)
y = np.random.randn(8)


# In[137]:


x


# In[138]:


y


# In[140]:


np.maximum(x, y)


# In[141]:


# maximum 比对两个数组，两个数组之间两个元素进行比较，大的则重新组合成一个数组


# In[142]:


arr = np.random.randn(7) * 5


# In[143]:


arr


# In[145]:


remainder, whole_part = np.modf(arr)


# In[146]:


remainder,


# In[147]:


whole_part


# In[148]:


# Ufuncs 可以接受一个 out 可选参数，这样的结果是直接在数组里进行操作
arr


# In[149]:


np.sqrt(arr)


# In[150]:


np.sqrt(arr, arr)


# In[151]:


arr


# ## 利用数组进行数据处理

# In[152]:


points = np.arange(-5, 5, 0.01)


# In[153]:


points


# In[154]:


xs, ys = np.meshgrid(points, points)


# In[155]:


xs


# In[156]:


ys


# In[157]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[158]:


z


# In[159]:


# 二维数组可视化使用 matplotlib
import matplotlib.pyplot as plt


# In[160]:


plt.imshow(z, cmap=plt.cm.gray);plt.colorbar()


# In[161]:


plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# In[162]:


plt.imshow(z, cmap=plt.cm.gray);plt.colorbar()


# ## 将条件逻辑表述为数组运算

# In[164]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[165]:


result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]


# In[166]:


result


# In[168]:


result = np.where(cond, xarr, yarr)


# In[169]:


result


# In[170]:


# 在数据分析工作中，where通常用于根据另一个数组而产生一个新的数组
arr = np.random.randn(4, 4)
arr


# In[171]:


arr > 0


# In[172]:


np.where(arr > 0, 2, -2)


# In[173]:


# 三参数：1.位置或表达式，最终是个表示位置的列表
#         2.为真时要替换的值
#         3.为假时要替换的值
np.where(arr>0, 2, arr)


# ## 数学和统计方法 

# In[174]:


arr = np.random.randn(5, 4)


# In[175]:


arr


# In[176]:


arr.mean()


# In[177]:


np.mean(arr)


# In[178]:


# 两种方法是取均值
arr.sum()  # 取元素和


# In[179]:


# aaxis 参数用于计算该轴向上的统计值
arr.mean(axis=1)


# In[180]:


arr.sum(axis=0)


# In[181]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# In[182]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


# In[183]:


arr


# In[184]:


arr.cumsum(axis=0)


# In[185]:


arr.cumsum(axis=0)


# In[186]:


arr.cumprod(axis=1)


# In[187]:


# cumsum 所有元素的累计和
# cumprod 所有元素的累计积


# ## 用于布尔弄数组的方法

# In[188]:


arr = np.random.randn(100)


# In[189]:


(arr > 0).sum()


# In[190]:


# any 测试数组中是否存在一个或多个 True
# all 则检查数组中所有值是否都是 True
bools = np.array([False, False, True, False])
bools.any()


# In[191]:


bools.all()


# ## 排序

# In[192]:


arr = np.random.randn(6)
print(arr)
arr.sort()
print(arr)


# In[196]:


## 多维数组可以在任何一个轴上进行排序，前提是将轴编号进行传递
arr = np.random.randn(5, 3)
print(arr)
arr.sort(1)


# In[197]:


arr


# In[198]:


# 顶级方法 np.sort 返回的是数组的已排序副本，而就地排序则会修改数组本身
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]


# ## 唯一化以及其它的集合逻辑
# 

# In[199]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[200]:


ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[201]:


values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# In[202]:


# np.in1d 测试一个数组中的值在另一个数组中的成员资格


# ## 用于数组的文件输入输出

# In[203]:


arr = np.arange(10)


# In[204]:


np.save('some_array', arr)


# In[205]:


get_ipython().run_line_magic('pwd', '')


# In[206]:


np.load('some_array.npy')


# In[207]:


np.savez('array_archive.npz', a=arr, b=arr)


# In[208]:


# np.save 保存为文件 
# np.savez 保存为压缩包，以关键字参数传入


# In[209]:


arch = np.load('array_archive.npz')


# In[210]:


arch


# In[211]:


arch['b']


# In[212]:


# 压缩， 则要上 np.savez_compressed
np.savez_compressed('array_compressed.npz', a=arr, b=arr)


# ## 线性代数

# In[213]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x, y


# In[214]:


x


# In[215]:


y


# In[216]:


x.dot(y)


# In[217]:


np.dot(x, np.ones(3))


# In[218]:


# @ 符号用中缀运算符，进行矩阵乘法
x @ np.ones(3)


# In[219]:


from numpy.linalg import inv, qr


# In[220]:


X = np.random.randn(5, 5)
mat = X.T.dot(X)


# In[221]:


inv(mat)


# In[222]:


mat.dot(inv(mat))


# In[223]:


q, r = qr(mat)


# In[224]:


r


# ## 伪随机数生成 

# In[226]:


samples = np.random.normal(size=(4, 4))
samples


# In[229]:


# python 内置 random 与 numpy.random 的速度对比
from random import normalvariate
N = 1000000


# In[230]:


get_ipython().run_line_magic('timeit', 'samples = [normalvariate(0, 1) for _ in range(N)]')


# In[231]:


get_ipython().run_line_magic('timeit', 'np.random.normal(size=N)')


# In[232]:


np.random.seed(1234)
# numpy.random 的数据生成使用全局的随机种子
# 要避免全局状态，使用 np.random.Random.State，创建一个与其它隔离的随机数生成器
rng = np.random.RandomState(1234)


# In[233]:


rng.randn(10)


# ## 示例：随机漫步

# In[234]:


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


# In[235]:


plt.plot(walk[:100])


# In[236]:


plt.plot(walk[:1000])


# In[238]:


nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws>0, 1, -1)
walk = steps.cumsum()


# In[239]:


walk.min()


# In[240]:


walk.max()


# In[241]:


(np.abs(walk)>=10).argmax()


# ## 一次模拟多个随机漫步

# In[242]:


nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws>0, 1, -1)
walks = steps.cumsum(1)
print(walks)
# 计算所有随机漫步的最大值，最小值
print('===========================')
print(walks.max())
print(walks.min())
print('===========================')
# 计算30或-30的最小穿越时间
print('===========================')
hits30 = (np.abs(walks)>=30).any(1)
print(hits30)
print(hits30.sum())
print('===========================')


# In[243]:


crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()

