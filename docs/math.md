# 数学

包含质数、约数、欧拉函数、快速幂、龟速乘、扩展欧几里得、中国剩余定理、高斯消元、求组合数、容斥原理、博弈论

## 质数

### 试除法判定

复杂度 $O(\sqrt{n})$

```cpp
bool is_prime(int n){
    if(n < 2) return false;
    for(int i = 2; i <= n/i; i++)
        if(n % i == 0) return false;
    return true;
}
```

### 分解质因数

复杂度 $O(logn) $ ~ $ O(\sqrt{n})$

```cpp
unordered_map<int, int> mp;

void divide(int n){
    for(int i = 2; i<= n/i; i++){
        while(n%i == 0) n /= i, mp[i]++;
    }
    if(n > 1) mp[n]++;
}
```

### 埃氏筛

复杂度 $O(nloglogn)$

```cpp
const int maxn = 1e6+10;
int primes[maxn], cnt;
bool vis[maxn];

void get_primes(int n){
    for(int i = 2; i<=n; i++){
        if(vis[i]) continue;
        primes[cnt++] = i;
        for(int j = i + i; j<=n; j += i){
            vis[j] = true;
        }
    }
}
```

### 线性筛

复杂度 $O(n)$

```cpp
const int maxn = 1e6+10;
int primes[maxn], cnt;
bool vis[maxn];

void get_primes(int n){
    for(int i = 2; i<=n; i++){
        if(!vis[i]) primes[cnt++] = i;
        for(int j = 0; primes[j] <= n/i; j++){
            vis[primes[j] * i] = true;
            if(i % primes[j] == 0) break;
        }
    }
}
```

## 约数

int范围内的整数，约数最多的数的约数大约有1500个

一个数可以质因数分解为 $p_1^{\alpha_1} \times p_2^{\alpha_2} \times\cdots\times p_k^{\alpha_k}$

### 试除法求所有约数

复杂度 $O(\sqrt{n})$

```cpp
vector<int> get_divisors(int n){
    
    vector<int> ret;
    
    for(int i = 1; i<= n/i; i++){
        if(n % i == 0) {
            ret.push_back(i);
            if(i != n/i) ret.push_back(n/i);
        }
    }
    
    sort(ret.begin(), ret.end());
    return ret;
}
```

### 约数个数

n个数的乘积的约数个数。将所有数因式分解，再套用下面公式。

$(\alpha_1 + 1)(\alpha_2 + 1)\cdots(\alpha_k + 1)$

```cpp
const int mod = 1e9+7;
unordered_map<int, int> mp;
 
void divide(int n){
    for(int i = 2; i<= n/i; i++){
        while(n%i == 0) n /= i, mp[i]++;
    }
    if(n > 1) mp[n]++;
}

int solve(){
    long long ans = 1;
    for(auto p : mp){
        ans = ans * (p.second + 1) % mod;
    }
    return ans;
}
```

### 约数之和

n个数的乘积的约数之和。将所有数因式分解，再套用下面公式。

$(p_1^0 + p_1^1+···+p_1^{\alpha_1})\cdots(p_k^0 + p_k^1+···+p_k^{\alpha_k})$

```cpp
const int mod = 1e9+7;
unordered_map<int, int> mp;

void divide(int n){
    for(int i = 2; i<= n/i; i++){
        while(n%i == 0) n /= i, mp[i]++;
    }
    if(n > 1) mp[n]++;
}

int solve(){
    long long ans = 1;
    for(auto p : mp){
        long long ret = 1;
        int a = p.first; int k = p.second;
        while(k--) ret = (ret*a+1) % mod;
        ans = ans * ret % mod;
    }
    return ans;
}
```

### 欧几里德

复杂度 $O(logn)$

```cpp
int gcd(int a, int b){
    return b ?  gcd(b, a % b) : a;
}
```

## 欧拉函数

### 单个数

瓶颈是分解质因数，复杂度 $O(\sqrt{n})$

先进行质因数分解，$\varphi(N) = N(1-\frac{1}{p_1})(1-\frac{1}{p_2})\cdots(1-\frac{1}{p_k})$

```cpp
int euler(int n){
    int ret = n;
    for(int i = 2; i<=n/i; i++ ){
        if(n%i == 0){
            ret -= ret/i;
            while(n%i == 0) n /= i;
        }
    }
    if(n > 1) ret -= ret/n;
    return ret;
}
```

### 筛法

复杂度 $O(n)$

```cpp
const int maxn = 1e6+20;

int primes[maxn], cnt;
int euler[maxn];
bool vis[maxn];

void get_eulers(int n){
    
    euler[1] = 1;
    for(int i = 2; i<=n; i++){
        if(!vis[i]){
            primes[cnt++] = i;
            euler[i] = i-1;
        }
        for(int j = 0; primes[j] <= n/i; j++){
            vis[primes[j] * i] = true;
            if(i % primes[j] == 0){
                euler[primes[j] * i] = euler[i] * primes[j];
                break;
            }
            euler[primes[j] * i] = euler[i] * (primes[j] - 1);
        }
    }
    
}
```

### 欧拉定理

若 $a$ 与 $n $ 互质，则 $a^{\varphi(n)}\equiv1\pmod{n}$

当 $n$ 为质数时，得到费马定理 $a^{p-1} \equiv1\pmod{p}$

## 快速幂

### 快速幂

```cpp
int qmi(int a, int k){
    int res = 1;
    while(k) {
        if(k & 1) res = (ll)res * a % mod;
        k >>= 1;
        a = (ll) a * a % mod;
    }
    return res;
}
```

### 快速幂求逆元

前提条件：模数为质数

```cpp
int inv(int a){
    if(a % mod == 0) return -1;
    else return qmi(a, mod-2);
}
```

## 龟速乘

```cpp
ll qadd(ll a, ll b, ll p){
    ll res = 0;
    while(b){
        if(b & 1) res = (res + a) % p;
        a = (a + a) % p;
        b >>= 1;
    }
    return res;
}
```

## 扩展欧几里得

```cpp
int exgcd(int a, int b, int &x, int &y)
{
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```

## 中国剩余定理

已知 $m_1,m_2,\cdots,m_k$ 两两互质。令 $M={m_1}\times{m_2}\times\cdots\times{m_l}$， $M_i=\frac{M}{m_i}$， $M_i^{-1}$ 表示 $M_i$ 模 $m_i$ 的逆。

对于线性同余方程组
$$
\begin{equation}
\left\{
\begin{aligned}
x\equiv{a_1}&\pmod{m_1}\\
x\equiv{a_2}&\pmod{m_2}\\
&\cdots\\
x\equiv{a_k}&\pmod{m_k}\\
\end{aligned}
\right.
\end{equation}
$$
存在解 $x=a_1M_1M_1^{-1}+a_2M_2M_2^{-1}+\cdots+a_{k}M_{k}M_{k}^{-1}$

### 朴素CRT

```cpp
typedef long long LL;
const int maxn = 12;

int n;
int a[maxn], m[maxn];

LL exgcd(LL a, LL b, LL &x, LL &y)
{
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

LL crt(){
    
    LL M = 1;
    for(int i = 0; i<n; i++){
        M *= m[i];
    }
    
    LL ans = 0;
    for(int i = 0; i<n; i++){
        LL inv, x;
        exgcd(M/m[i], m[i], inv, x);
        ans += a[i] * M/m[i] * inv;
    }
    
    return (ans % M + M) % M;
}
```

### 扩展CRT

可以在 $m_i$ 与 $m_j$ 不互质的情况下进行求解，无解时返回-1

```cpp
typedef long long LL;
const int maxn = 30;

int n;
int a[maxn], m[maxn];

int exgcd(LL a, LL b, LL &x, LL &y)
{
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

LL excrt(){
    
    LL x = 0, m1 = m[0], a1 = a[0];
    for (int i = 1; i < n ; i ++ ) {
        LL m2 = m[i], a2 = a[i];
        LL k1, k2;
        LL d = exgcd(m1, m2, k1, k2);
        if ((a2 - a1) % d) return -1;

        k1 *= (a2 - a1) / d;
        k1 = (k1 % (m2/d) + m2/d) % (m2/d);

        x = k1 * m1 + a1;

        LL m = abs(m1 / d * m2);
        a1 = k1 * m1 + a1;
        m1 = m;
    }

    if (x != -1) x = (a1 % m1 + m1) % m1;

    return x;
}
```

## 高斯消元

复杂度 $O(n^{3})$

### 朴素做法

```cpp
const int N = 110;
const double eps = 1e-6;

int n;
double a[N][N];

//0.有唯一解 1.有无穷多解 2.无解
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][c]) > fabs(a[t][c]))
                t = i;

        if (fabs(a[t][c]) < eps) continue;

        for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);
        for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];

        for (int i = r + 1; i < n; i ++ )
            if (fabs(a[i][c]) > eps)
                for (int j = n; j >= c; j -- )
                    a[i][j] -= a[r][j] * a[i][c];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (fabs(a[i][n]) > eps)
                return 2;
        return 1;
    }

    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] -= a[j][n] * a[i][j];

    return 0;
}
```

### 01异或线性方程组

```cpp
const int N = 110;

int n;
int a[N][N];

//0.有唯一解 1.有无穷多解 2.无解
int gauss()
{
    int c, r;
    for (c = 0, r = 0; c < n; c ++ )
    {
        int t = r;
        for (int i = r; i < n; i ++ )
            if (a[i][c])
                t = i;

        if (!a[t][c]) continue;

        for (int i = c; i <= n; i ++ ) swap(a[r][i], a[t][i]);
        for (int i = r + 1; i < n; i ++ )
            if (a[i][c])
                for (int j = n; j >= c; j -- )
                    a[i][j] ^= a[r][j];

        r ++ ;
    }

    if (r < n)
    {
        for (int i = r; i < n; i ++ )
            if (a[i][n])
                return 2;
        return 1;
    }

    for (int i = n - 1; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            a[i][n] ^= a[i][j] & a[j][n];

    return 0;
}
```

## 求组合数

### 递推

$1 \leq b \leq a \leq 2000$ 复杂度 $O(n^{2})$

```cpp
const int maxn = 2010;
const int mod = 1e9+7;

int c[maxn][maxn];

void init(){
    for(int i = 0; i<maxn; i++){
        for(int j = 0; j<=i; j++){
            if(!j) c[i][j] = 1;
            else c[i][j] = (c[i-1][j] + c[i-1][j-1]) % mod;
        }
    }
}
```

### 预处理

$1 \leq b \leq a \leq {10}^{5}$  复杂度 $O(nlogn)$

```cpp
typedef long long LL;

const int mod = 1e9+7;
const int maxab = 1e5+20;

int fact[maxab];
int infact[maxab];

int qmi(int a, int k, int p){
    int res = 1;
    while(k) {
        if(k & 1) res = (LL)res * a % p;
        k >>= 1;
        a = (LL) a * a % p;
    }
    return res;
}

void init(){
    fact[0] = infact[0] = 1;
    for(int i = 1; i<maxab; i++){
        fact[i] = (LL)fact[i-1] * i % mod;
        infact[i] = (LL)infact[i-1] * qmi(i, mod-2, mod) % mod;
    }
}

int C(int a, int b){
    return (LL)fact[a] * infact[b] % mod * infact[a-b] % mod;
}
```

### 卢卡斯定理

$1 \leq b \leq a \leq {10}^{18}$ , $1 \leq p \leq 10^{5}$  复杂度 $O(plogNlogp)$

```cpp
typedef long long LL;

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int C(int a, int b, int p) {
    if (b > a) return 0;

    int res = 1;
    for (int i = 1, j = a; i <= b; i ++, j -- )
    {
        res = (LL)res * j % p;
        res = (LL)res * qmi(i, p - 2, p) % p;
    }
    return res;
}

int lucas(LL a, LL b, int p) {
    if (a < p && b < p) return C(a, b, p);
    return (LL)C(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}
```

### 高精度

分解质因数，再进行高精度乘法

```cpp
const int maxab = 5010;

int primes[maxab], cnt;
int sum[maxab];
bool st[maxab];


void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}

//获取n!中含有多少个因子p
int get(int n, int p)
{
    int res = 0;
    while (n)
    {
        res += n / p;
        n /= p;
    }
    return res;
}

//高精乘
vector<int> mul(vector<int> a, int b)
{
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size(); i ++ ) {
        t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    while (t)
    {
        c.push_back(t % 10);
        t /= 10;
    }
    return c;
}

void init(){
    get_primes(maxab);
}

//注意:反序
vector<int> C(int a, int b){
    for (int i = 0; i < cnt; i ++ ) {
        int p = primes[i];
        sum[i] = get(a, p) - get(a - b, p) - get(b, p);
    }
    
    vector<int> res(1, 1);
    
    for (int i = 0; i < cnt; i ++ )
        for (int j = 0; j < sum[i]; j ++ )
            res = mul(res, primes[i]);
    
    return res;
}
```

## 卡特兰数

通项: $C_n=\frac{C_{2n}^{n}}{n+1}$

递推式: $C_1=1, C_n=\frac{4n-2}{n+1}C_{n-1} $

### 单次

将所有阶乘预处理出来

```cpp
typedef long long LL;

const int mod = 1e9+7;
const int maxn = 2e5+20;

int fact[maxn];
int infact[maxn];

int qmi(int a, int k, int p){
    int res = 1;
    while(k) {
        if(k & 1) res = (LL)res * a % p;
        k >>= 1;
        a = (LL) a * a % p;
    }
    return res;
}

void init(){
    fact[0] = infact[0] = 1;
    for(int i = 1; i<maxn; i++){
        fact[i] = (LL)fact[i-1] * i % mod;
        infact[i] = (LL)infact[i-1] * qmi(i, mod-2, mod) % mod;
    }
}

int C(int a, int b){
    return (LL)fact[a] * infact[b] % mod * infact[a-b] % mod;
}

int catalan(int n){
    return (LL)C(2*n, n) * qmi(n+1, mod-2, mod) % mod;
}

```

### 递推

```cpp
typedef long long LL;

const int mod = 1e9 + 7;
const int maxn = 1e5+10;

int c[maxn]; //卡特兰数

int qmi(int a, int k, int p) {
    int res = 1;
    while (k) {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

void get_catalans(int n, int p){
    c[1] = 1;
    for(int i = 2; i<=n; i++){
        c[i] = (LL)(4*i-2) * c[i-1] % p * qmi(i+1, p-2, p) % p;
    }
}
```

## 容斥原理

给定一个整数 $n$ 和 $m$ 个不同的质数 $p_1, p_2, \cdots, p_m$

求出 $1$ ~ $n$ 中能被 $p_1, p_2, \cdots, p_m$ 中的至少一个数整除的整数有多少个。

复杂度 $O(2^m)$

```cpp
const int maxn = 20;
int n, m;
int p[maxn];

int solve(){
    
    int ans = 0;
    
    for(int i = 1; i< 1<<m; i++){ //状压 枚举所有选择情况
        
        int t = 1, cnt = 0;
        
        for(int j = 0; j<m; j++){
            if(i>>j & 1){
                if((long long) t * p[j] > n){t = -1; break;}
                t *= p[j];
                cnt++;
            }
        }
        
        if(t != -1){
            if(cnt & 1) ans += n/t;
            else ans -= n/t;
        }
        
    }
    
    return ans;
}
```

## 博弈论

### sg函数

$sg(必败态) = 0$ 

$sg(x) = MEX(y_1, y_2, \cdots, y_n)$

对于可以转化为多个状态的博弈，若 $sg(x_1) \oplus sg(x_2) \oplus \cdots\oplus sg(x_n) = 0$ 则必败，否则必胜

```cpp
int f[120];

int sg(int x){
    
    if(f[x] != -1) return f[x];
    
    unordered_set<int> S;
    
    //计算当前状态可以转化到的所有sg值
    for(int i = 0; i<x; i++){
        for(int j = 0; j<x; j++){
            S.insert(sg(i) ^ sg(j));
        }
    }
    
    for(int i = 0; ; i++){
        if(!S.count(i)) return f[x] = i;
    }
}
```

### Nim博弈

$a_1 \oplus a_2\oplus\cdots\oplus a_n = 0$ 则必败，否则必胜

​	
