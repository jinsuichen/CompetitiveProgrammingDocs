# 关于

包含邻接表、树的重心、有向图的拓扑排序、Dijkstra、bellman-ford、spfa、Floyd、Prim、Fruskal、染色法判定二分图、匈牙利算法。

# 邻接表

```cpp
const int maxn = 1e5+50; //点的数量
const int maxm = maxn*2; //边的数量

int h[maxn], e[maxm], ne[maxm];
bool vis[maxn];
int top;

void init(){
    memset(h, -1, sizeof(h));
}

void add_edge(int a, int b){
    e[top] = b;
    ne[top] = h[a];
    h[a] = top++;
}

void travel(int u)[
    for(int i = h[u]; i!=-1; i = ne[i]){
        int v = e[i];
        //do something
    }
]
```

# 树的重心

重心定义：重心是指树中的一个结点，如果将这个点删除后，剩余各个连通块中点数的最大值最小，那么这个节点被称为树的重心。

```cpp
int ans = 0x3f3f3f3f;
int n; //点数

//返回以某结点为根的z
int dfs(int u){
    
    vis[u] = true;
    int res = 0, tot = 1;
    for(int i = h[u]; i!=-1; i = ne[i]){
        int v = e[i];
        if(!vis[v]){
            int p = dfs(v);
            res = max(res, p);
            tot += p;
        }
    }
    res = max(res, n - tot);
    ans = min(ans, res);
    
    return tot;
}

void solve(){
    dfs(1);
}
```

# 拓扑排序

```cpp
int n; //结点数
int in[maxn]; //入度

bool topsort(vector<int>& ans){
    ans.clear();
    int cnt = 0; //插入结点的数量
    
    queue<int> q;
    for(int i = 1; i<=n; i++)
        if(!in[i]) {
            q.push(i);
            ans.push_back(i);
            cnt++;
        }
            
    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int i = h[u]; i!=-1; i = ne[i]){
            int v = e[i];
            if(--in[v] == 0) {
                q.push(v);
                ans.push_back(v);
                cnt++;
            }
        }
    }
    
    return cnt == n;
}
```

# 最短路算法选择

```mermaid
graph LR
A(最短路)
B(单源最短路)
C(多源汇最短路)
D(所有边权都是正数)
E(存在负边权)
F("朴素Dijkstra O(n^2)")
G("堆优化Dijkstra O(mlogn)")
H("Bellman-Ford O(nm)")
I("SPFA 一般O(m) 最坏O(nm)")
J("Floyd O(n^3)")
A-->B
A-->C
C--->J
B-->D
B-->E
D-->F
D-->G
E-->H
E-->I
```

# Dijkstra

## 朴素做法

```cpp
int n, m;
const int maxn = 520;
const int INF = 0x3f3f3f3f;
int g[maxn][maxn];
int dist[maxn];
bool vis[maxn];

void init(){
    memset(g, 0x3f, sizeof(g));
}

//需要处理重边
void add_edge(int a, int b, int val){
    g[a][b] = min(g[a][b], val);
}

//不连通返回-1
int dijkstra(){
    memset(dist, 0x3f, sizeof(dist));
    dist[1] = 0;
    
    for(int _ = 0; _ < n-1; _++){
        int t = -1;
        for(int i = 1; i<=n; i++)
            if(!vis[i] && (t==-1 || dist[i] < dist[t]))
                t = i;
        vis[t] = true;
        for(int i = 1; i<=n; i++)
            dist[i] = min(dist[i], dist[t] + g[t][i]);
    }
    
    return dist[n] == INF ? -1 : dist[n];
}
```

## 堆优化

```cpp
typedef pair<int, int> PII;

const int maxn = 2e5+50;
const int INF = 0x3f3f3f3f;
int n, m;

int dist[maxn];
bool vis[maxn];

int h[maxn], e[maxn], weight[maxn], ne[maxn], idx;

void init(){
    memset(h, -1, sizeof(h));
}

void add_edge(int a, int b, int w){
    e[idx] = b; weight[idx] = w; 
    ne[idx] = h[a];
    h[a] = idx++;
}

//不连通返回-1
int dijkstra(){

    memset(dist, 0x3f, sizeof(dist));
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII> > q;
    q.push({0, 1});

    while(!q.empty()){
        PII t = q.top(); q.pop();

        int dis = t.first; int u = t.second;
        if(vis[u]) continue;
        vis[u] = true;

        for(int i = h[u]; i!=-1; i = ne[i]){
            int v = e[i], w = weight[i];
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                q.push({dist[v], v});
            }
        }

    }

    return dist[n] == INF ? -1 : dist[n];
}
```

# bellman-ford

1 号点到 n 号点的最多经过 k 条边的最短距离

```cpp
const int maxn = 510;
const int maxm = 1e4+20;
const int INF = 0x3f3f3f3f;

int n, m, k;
int dist[maxn], backup[maxn];
int a[maxm], b[maxm], w[maxm]; //存边

//不连通返回INF
int bellman_ford(){
    
    memset(dist, 0x3f, sizeof(dist));
    dist[1] = 0;
    for(int i = 0; i<k; i++){        
        memcpy(backup, dist, sizeof(dist));
        for(int j = 0; j<m; j++){
            dist[b[j]] = min(dist[b[j]], w[j] + backup[a[j]]);
        }
    }    
    return dist[n] > INF / 2 ? INF : dist[n]; 
}
```

# spfa

## 求最短路

```cpp
const int maxn = 1e5+50;
const int INF = 0x3f3f3f3f;
int n, m;

int dist[maxn];
bool vis[maxn];

int h[maxn], e[maxn], weight[maxn], ne[maxn], idx;

void init(){
    memset(h, -1, sizeof(h));
}

void add_edge(int a, int b, int w){
    e[idx] = b; weight[idx] = w; 
    ne[idx] = h[a];
    h[a] = idx++;
}

//不连通返回INF
int spfa(){

    memset(dist, 0x3f, sizeof(dist));
    dist[1] = 0;
    
    queue<int> q; q.push(1); vis[1] = true;
    while(!q.empty()){
        int u = q.front(); q.pop(); vis[u] = false;
        
        for(int i = h[u]; i!=-1; i = ne[i]){
            int v = e[i], w = weight[i];
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                if(!vis[v]) q.push(v), vis[v] = true;
            }
        }
    }
    
    return dist[n];
}
```

## 判断负环

```cpp
const int maxn = 1e5+50;
const int INF = 0x3f3f3f3f;
int n, m;

int dist[maxn], cnt[maxn];
bool vis[maxn];

int h[maxn], e[maxn], weight[maxn], ne[maxn], idx;

void init(){
    memset(h, -1, sizeof(h));
}

void add_edge(int a, int b, int w){
    e[idx] = b; weight[idx] = w; 
    ne[idx] = h[a];
    h[a] = idx++;
}

bool spfa(){
    
    queue<int> q;
    for(int i = 1; i<=n; i++){
        q.push(i);
        vis[i] = true;
    }
    
    while(!q.empty()){
        int u = q.front(); q.pop(); vis[u] = false;
        
        for(int i = h[u]; i!=-1; i = ne[i]){
            int v = e[i], w = weight[i];
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                cnt[v] = cnt[u] + 1;
                if(cnt[v] >= n) return true;
                if(!vis[v]) q.push(v), vis[v] = true;
            }
        }
    }
    
    return false;
}
```

# Floyd

```cpp
int n, m, k;
const int INF = 0x3f3f3f3f;
const int maxn = 210;
int g[maxn][maxn];

void init(){
    memset(g, 0x3f, sizeof(g));
    for(int i = 1; i<=200; i++) g[i][i] = 0;
}

//需要处理重边
void add_edge(int a, int b, int w){
    g[a][b] = min(g[a][b], w);
}

//g[i][j] > INF / 2 表示不连通
void floyd(){
    
    for(int k = 1; k<=n; k++){
        for(int i = 1; i<=n; i++){
            for(int j = 1; j<=n; j++){
                g[i][j] = min(g[i][k] + g[k][j] , g[i][j]);
            }
        }
    }
    
}
```

# Prim

## 朴素做法

用于稠密图。复杂度 $O(n^2)$ 

```cpp
const int maxn = 520;
const int maxm = 1e5+20;
const int INF = 0x3f3f3f3f;

int n, m; 
int dist[maxm];
int g[maxn][maxn];
bool vis[maxn];

void init(){
    memset(g, INF, sizeof(g));
}

void add_edge(int a, int b, int v){
     g[a][b] = g[b][a] = min(g[a][b], v);
}

// 不连通时返回INF
int prim(){
    
    memset(dist, INF, sizeof(dist));
    dist[1] = 0;
    
    int ret = 0;
    for(int i = 0; i<n; i++){
    
        int t = -1;
        for(int j = 1; j<=n; j++) 
            if(!vis[j] && (t == -1 || (dist[j] < dist[t])))
                t = j;
        
        if(dist[t] == INF) return INF;
        ret += dist[t];
        
        for(int j = 1; j<=n; j++)
            dist[j] = min(dist[j], g[t][j]);
            
        vis[t] = true;
    }
    
    return ret;
}
```

## 堆优化

用于稀疏图，不常用。复杂度 $O(mlogn)$ 

# Kruskal

用于稀疏图。复杂度 $O(mlogm)$

```cpp
const int INF = 0x3f3f3f3f;
const int maxn = 1e5+20;
const int maxm = 2e5+20;

int n, m;
struct Edge{
    int u, v, w;
    bool operator < (const Edge& e) const {
        return w < e.w;
    };
};
Edge edges[maxm];

int p[maxm];
int find(int x){
    if(x != p[x]) p[x] = find(p[x]);
    return p[x];
}

// 不连通时返回INF
int kruskal(){
    
    for(int i = 1; i<=n; i++) p[i] = i;
    
    int res = 0;
    int cnt = 0;
    
    sort(edges, edges+m);
    for(int i = 0; i<m; i++){
        int a = edges[i].u, b = edges[i].v, w = edges[i].w;
        int pa = find(a); int pb = find(b);       
        if(pa != pb){
            p[pa] = pb;
            res += w;
            cnt++;
        }
    }
    
    if(cnt == n-1) return res;
    else return INF;
}
```

# 染色法判定二分图

复杂度 $O(n+m)$

一个图是二分图当且仅当图中不含有奇数环。

```cpp
const int maxn = 1e5+20;
const int maxm = 2e5+20;

int n, m;
int h[maxn], e[maxm], ne[maxm]; int top;
int color[maxn];

void init(){
    memset(color, -1, sizeof(color));
    memset(h, -1, sizeof(h));
}

void add_edge(int u, int v){
    e[top] = v;
    ne[top] = h[u];
    h[u] = top++;
}

bool dfs(int u, int c){
    
    if(color[u] != -1) return color[u] == c;
    
    color[u] = c;
    for(int i = h[u]; i!=-1; i = ne[i]){
        if(!dfs(e[i], c^1)) 
            return false;
    }
    return true;
}

bool solve(){
    
    for(int i = 1; i<=n; i++){
        if(color[i] == -1) 
            if(!dfs(i, 0))
                return false;
    }
    return true;
}
```

# 匈牙利算法

复杂度 $O(nm)$ ，但实际运行时间一般远小于 $O(nm)$ 。

```cpp
const int maxn = 520;
const int maxm = 1e5+20;

int n1, n2, m;
int h[maxn], ne[maxm], e[maxm], top;
bool vis[maxn];
int match[maxn];

void init(){
    memset(h, -1, sizeof(h));
}

void add_edge(int a, int b){
    e[top] = b;
    ne[top] = h[a];
    h[a] = top++;
}

bool find(int x){
    
    for(int i = h[x]; i!=-1; i = ne[i]){
        int j = e[i];
        if(vis[j]) continue;
        vis[j] = true;
        if(match[j] == 0 || find(match[j])){
            match[j] = x;
            return true;
        }
    }
    
    return false;
}

int solve(){
    int res = 0;
    
    for(int i = 1; i<=n1; i++){
        memset(vis, 0, sizeof(vis));
        if(find(i)) res++;
    }
    
    return res;
}
```

