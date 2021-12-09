# 关于

包含单调栈、单调队列、KMP、Trie、并查集、堆、哈希表、字符串哈希。

# 单调栈

输出每个数左边第一个比它小的数，如果不存在则输出 −1

```cpp
const int maxn = 1e5+10;
int a[maxn];
stack<int> s;

void solve(){
    for(int i = 0; i<n; i++){
        while(!s.empty() && s.top() >= a[i]) s.pop();
        if(s.empty()) printf("-1 ");
        else printf("%d ", s.top());
        s.push(a[i]);
    }
}
```

# 单调队列

滑动窗口，输出最小值和最大值

```cpp
int n, k; 
const int maxn = 1000100;
int a[maxn];
deque<int> q;

void solve1(){
    q.clear();
    for(int i = 0; i<n; i++){
        if(!q.empty() && i - k + 1 > q.front()) q.pop_front();
        while(!q.empty() && a[i] <= a[q.back()]) q.pop_back();
        q.push_back(i);
        if(i >= k-1) printf("%d ", a[q.front()]);
    }
}

void solve2(){
    q.clear();
    for(int i = 0; i<n; i++){
        if(!q.empty() && i - k + 1 > q.front()) q.pop_front();
        while(!q.empty() && a[i] >= a[q.back()]) q.pop_back();
        q.push_back(i);
        if(i >= k-1) printf("%d ", a[q.front()]);
    }
}
```

# KMP

```cpp
int n, m; //n代表原串长度，m代表模式串长度
char s[1000010]; //原串，从1开始存储
char p[100010]; //模式串，从1开始存储
int nxt[100010];

void init(){
    cin >> m >> p+1 >> n >> s+1;
    for(int i = 2, j = 0; i<=m; i++){
        while(j && p[i] != p[j+1]) j = nxt[j];
        if(p[i] == p[j+1]) j++;
        nxt[i] = j;
    }
}

void solve(){
    for(int i = 1, j = 0; i<=n; i++){
        while(j && s[i] != p[j+1]) j = nxt[j];
        if(s[i] == p[j+1]) j++;
        if(j == m){
            printf("%d ", i-m); // i-m 为模式串在原串中的起始下标
            //do something
            j = nxt[j];
        }
    }
}
```

# Trie

```cpp
const int maxn = 1e5+50;

int cnt[maxn];
int son[maxn][26];
int idx;

char str[maxn];

void insert(char * s){
    int r = 0;
    for(int i = 0; s[i]; i++){
        int u = s[i] - '0';
        if(!son[r][u]) son[r][u] = ++idx;
        r = son[r][u];
    }
    cnt[r]++;
}

int query(char* s){
    int r = 0;
    for(int i = 0; s[i]; i++){
        int u = s[i] - '0';
        if(!son[r][u]) return 0;
        r = son[r][u];
    }
    return cnt[r];
}
```

# 并查集

```cpp
const int maxn = 1e5+50;

int p[maxn];

int find(int x){
    if(p[x] != x) p[x] = find(p[x]);
    return p[x];
}

void merge(int a, int b){
    p[find(a)] = find(b);
}

bool connected(int a, int b){
    return find(a) == find(b);
}
```

# 堆

## 快速建立

快速建立的前提条件是已知所有所有数据。

```cpp
for (int i = 1; i <= n; i ++ ) scanf("%d", &h[i]);
cnt = n;
for (int i = n / 2; i; i -- ) down(i);
```

## 简单堆

实现插入、查询最值、弹出最值

```cpp
const int maxn = 1e6+50;
int h[maxn];
int sz = 0;

void up(int p){
    while(p>>1 && h[p] < h[p>>1]){
        swap(h[p], h[p>>1]);
        p >>= 1;
    }
}

void down(int p){
    int t = p;
    if(p<<1 <= sz && h[p<<1] < h[t]) t = p<<1;
    if((p<<1|1) <= sz && h[p<<1|1] < h[t]) t = p<<1|1;
    if(t != p) {
        swap(h[p], h[t]);
        down(t);
    }
}

void insert(int x){
    h[++sz] = x;
    up(sz);
}

int top(){
    return h[1];
}

void pop(){
    h[1] = h[sz--];
    down(1);
}
```

## 完全堆

实现插入、查询最值、弹出最值、删除与修改第k个插入的数

```cpp
const int maxn = 1e6+50;
int h[maxn];
int ph[maxn];
int hp[maxn];
int sz = 0;
int cnt = 0;

void heap_swap(int a, int b){
    swap(ph[hp[a]],ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void up(int p){
    while(p>>1 && h[p] < h[p>>1]){
        heap_swap(p, p>>1);
        p >>= 1;
    }
}

void down(int p){
    int t = p;
    if(p<<1 <= sz && h[p<<1] < h[t]) t = p<<1;
    if((p<<1|1) <= sz && h[p<<1|1] < h[t]) t = p<<1|1;
    if(t != p) {
        heap_swap(p, t);
        down(t);
    }
}

void insert(int x){
    h[++sz] = x;
    ph[++cnt] = sz;
    hp[sz] = cnt;
    up(sz);
}

int top(){
    return h[1];
}

void pop(){
    heap_swap(1, sz--);
    down(1);
}

void delete_k(int k){
    k = ph[k];
    heap_swap(k, sz);
    sz--;
    up(k);
    down(k);
}

void update_k(int k, int x){
    k = ph[k];
    h[k] = x;
    up(k);
    down(k);
}
```

# 哈希表

## 常用素数

```hh
1e4+7		1e5+3		1e6+3		1e7+19		1e8+7
2e4+11		2e5+3		2e6+3		2e7+3		2e8+33
3e4+11		3e5+7		3e6+17		3e7+1		3e8+7
```

## 拉链法

要将h数组初始化为-1

maxn为一个大于数据范围的最小质数。<del>尽量令这个质数尽量远离2的整数倍。</del>

```cpp
const int maxn = 1e5+3;

int h[maxn];
int val[maxn], nxt[maxn];
int top = 0;

void init(){
    memset(h, -1, sizeof(h));
}

void insert(int x){
    int k = (x % maxn + maxn) % maxn; //hash
    val[top] = x;
    nxt[top] = h[k];
    h[k] = top;
    top++;
}

bool query(int x){
    int k = (x % maxn + maxn) % maxn; //hash
    for(int i = h[k]; i != -1; i = nxt[i])
        if(val[i] == x)
            return true;
    return false;
}
```

## 开放寻址法

null为数据中不可能出现的一个值，要将h初始化为自己定义的null

maxn为一个大于数据范围2-3倍的最小质数。<del>尽量令这个质数尽量远离2的整数倍。</del>

```find()``` 函数：如果找到k返回它的下标，如果没找到，返回它应该存在的位置。

```cpp
const int maxn = 2e5+3;
const int null = 0x3f3f3f3f;

int h[maxn];

void init(){
    memset(h, 0x3f, sizeof(h));
}

int find(int x){
    int k = (x % maxn + maxn) % maxn; //hash
    while(h[k] != null && h[k] != x){
        k++;
        if(k == maxn) k = 0;
    }
    return k;
}

void insert(int x){
    int k = find(x);
    h[k] = x;
}

bool query(int x){
    int k = find(x);
    return h[k] == x;
}
```

# 字符串哈希

p取 $131$ 或 $13331$ ，mod取 $2^{64}$ 时，冲突概率最小

```cpp
const int maxn = 1e5+50;

unsigned long long h[maxn];
unsigned long long p[maxn];

int n, m; 
char str[maxn];

void init(){
    p[0] = 1;
    for(int i = 1; i<=n; i++){
        h[i] = h[i-1] * 131 + str[i-1];
        p[i] = p[i-1] * 131;
    }
}

int get_hash(int l, int r){
    return h[r] - h[l-1] * p[r-l+1];
}
```
