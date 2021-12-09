# vector

vector按照字典序比较

```cpp
vector <int> a; //定义一个vector数组a
vector <int> a(10); //定义一个长度为10的vector数组a
vector <int> a(10,3); //定义一个长度为10的vector数组a，并且所有元素都为3
```

```cpp
size(); //返回元素个数
empty(); //返回是否是空
clear(); //清空
front(); //返回vector的第一个数
back(); //返回vector的最后一个数
push_back(); //向vector的最后插入一个数
pop_back(); //把vector的最后一个数删掉
begin(); //vector的第0个数
end(); //vector的最后一个的数的后面一个数
```

# pair

支持比较运算，以first为第一关键字，以second为第二关键字

```cpp
first(); //第一个元素
second(); //第二个元素
```

# string

```cpp
substr(); //返回每一个子串
c_str(); //返回这个string对应的字符数组的头指针
size(); //返回字母个数
length(); //返回字母个数
empty(); //返回字符串是否为空
clear(); //把字符串清空
```

# queue

```cpp
size(); //这个队列的长度
empty(); //返回这个队列是否为空
push(); //往队尾插入一个元素
front(); //返回队头元素
back(); //返回队尾元素
pop(); //把队头弹出
q = queue<int> (); //队列没有clear()函数
```

# priority_queue

大根堆：priority_queue <类型> 变量名;
小根堆：priority_queue <类型, vecotr <类型>, greater <类型> > 变量名

```cpp
size(); //这个堆的长度
empty(); //返回这个堆是否为空
push();//往堆里插入一个元素
top(); //返回堆顶元素
pop(); //弹出堆顶元素
//注意：堆没有clear函数！！！
```

# stack

```cpp
size(); //这个栈的长度
empty(); //返回这个栈是否为空
push(); //向栈顶插入一个元素
top(); //返回栈顶元素
pop(); //弹出栈顶元素
```

# deque

```cpp
size(); //这个双端队列的长度
empty(); //返回这个双端队列是否为空
clear(); //清空这个双端队列
front(); //返回第一个元素
back(); //返回最后一个元素
push_back(); //向最后插入一个元素
pop_back(); //弹出最后一个元素
push_front(); //向队首插入一个元素
pop_front(); //弹出第一个元素
begin(); //双端队列的第0个数
end(); //双端队列的最后一个的数的后面一个数
```

# set/multiset

 基于平衡二叉树（红黑树），动态维护有序序列

```cpp
size(); //返回元素个数
empty(); //返回set是否是空的
clear(); //清空
begin(); //第0个数，支持++或--，返回前驱和后继
end(); //最后一个的数的后面一个数，支持++或--，返回前驱和后继
insert(); //插入一个数
find(); //查找一个数
count(); //返回某一个数的个数
erase();
	//（1）输入是一个数x，删除所有x    O(k + log n)
	//（2）输入一个迭代器，删除这个迭代器
lower_bound(x); //返回大于等于x的最小的数的迭代器
upper_bound(x); //返回大于x的最小的数的迭代器
```

# map/multimap

 基于平衡二叉树（红黑树），动态维护有序序列

```cpp
insert(); //插入一个数，插入的数是一个pair
erase(); 
    //（1）输入是pair
    //（2）输入一个迭代器，删除这个迭代器
find(); //查找一个数
lower_bound(x); //返回大于等于x的最小的数的迭代器
upper_bound(x); //返回大于x的最小的数的迭代器
```

# unordered_xxx

基于哈希表

和上面类似，增删改查的时间复杂度是O(1)
不支持lower_bound()和upper_bound()

# bitset

bitset <位数> 变量名;

支持的运算符:    ~	&	|	^	>>	<<	==s	!=	[]

```cpp
count(); //返回某一个数的个数
any(); //判断是否至少有一个1
none(); //判断是否全为0
set(); //把所有位置赋值为1
set(k,v); //将第k位变成v
reset(); //把所有位变成0
flip(); //把所有位取反，等价于~
flip(k); //把第k位取反
```

