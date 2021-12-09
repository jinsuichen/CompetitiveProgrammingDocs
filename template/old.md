update on 10-22-2021

# 杂项

## 注意事项

1. 所有题都要考虑int和long long，不要无脑开long long。开long long时注意用```%lld```而不是```%d```。
2. 莫名WA考虑longlong和高精度。（可能数据不需要开longlong但过程或结果需要开longlong，此时要记得开longlong）
3. 永远都要多组输入
4. never never use cin & cout
5. 能用数组就不用STL
6. 适当删除头文件
7. 让main函数返回0

## 基础模板

```cpp
#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <deque>
#include <algorithm>
#include <stack>
#include <sstream>
#include <cstdlib>
#include <ctime>
/*-----------------------*/
//#define int long long
#define INF 0x3f3f3f3f
#define INF2 0x7f7f7f7f
using namespace std;
/*-----------------------*/
signed main() {

#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif



    

    return 0;

}
```



# C++STL

## 优先队列



## 双端队列

```deque<类型>d; ```

```push_back(x)/push_front(x)```//把x压入后/前端
```back()/front() ```//访问(不删除)后/前端元素
```pop_back() pop_front() ```//删除后/前端元素 
```empty()```//判断deque是否空 
```size()```//返回deque的元素数量 
```lear() ```//清空deque 

## pair头文件

```#include<utility>```

## getline用法

```cpp
#include <iostream>
#include <string>
using namespace std;

int main()
{
	string name;
	cout << "Please input your name: ";
	getline(cin, name);
	cout << "Welcome to here!" << name << endl;
	
	return 0;

}

```

## sstream用法

```cpp
    string str= "hello world I am very happy!";                           
    stringstream sstream(str);                                              //sstream<<
 
    while (sstream)
      {
        string substr;
 
        sstream>>substr;
        cout << substr << endl;    //也可vec.push_back(substr);
      } 
```

## 集合操作

​		set里面有```set_intersection```（取集合交集）、```set_union```（取集合并集）、```set_difference```（取集合差集）、```set_symmetric_difference```（取集合对称差集）等函数。其中，关于函数的五个参数问题做一下小结：

1. 这几个函数的前四个参数一样，只有第五个参数有多重版本。

2. EX1：```set_union(A.begin(),A.end(),B.begin(),B.end(),inserter(C, C.begin()));```前四个参数依次是第一的集合的头尾，第二个集合的头尾。第五个参数的意思是将集合A、B取合集后的结果存入集合C中。

3. EX2：```set_union(A.begin(),A.end(),B.begin(),B.end(),ostream_iterator<int>(cout," “));```这里的第五个参数的意思是将A、B取合集后的结果直接输出，```（cout," "）```双引号里面是输出你想用来间隔集合元素的符号或是空格。



## 输出全排列

```cpp
	int n, p[10];
    scanf("%d", &n);
    for(int i = 0; i<n; i++) scanf("%d", &p[i]);
    sort(p, p+n);
    do{
        for(int i = 0; i<n; i++) printf ("%d ", p[i]);
        puts("");
    }while(next_permutation(p, p+n));
```

```prev_permutation```为前一个全排列。

布尔型返回值，如果存在下一个/上一个全排列，则返回True，否则返回False。





# 数学 数论

## gcd lcm

最大公因数gcd
```cpp
int gcd(int a, int b){  // 一般要求a>=0, b>0。若a=b=0，代码也正确，返回0
	return b ? gcd(b, a%b):a;
}
```

最小公倍数lcm
```cpp
int gcd(int a, int b){  // 一般要求a>=0, b>0。若a=b=0，代码也正确，返回0
	return b ? gcd(b, a%b):a;
}

int lcm(int a, int b){ 
	return a / gcd(a, b) * b;
}
```



## ex_gcd（扩展欧几里德算法）

找出一对整数 $(x, y)$ 使得  $ax+by = gcd(a, b)$

d为a和b的最大公因数，x和y为一组特解

```cpp
void ex_gcd(int a, int b, int& d, int& x, int& y){
    if(!b) { d = a; x = 1; y = 0; }
    else{ ex_gcd(b, a%b, d, y, x); y -= x*(a/b); }
}
```

## 质数/素数

### Eratosthenes 筛法

注意，i为质数时，vis[i] == false

==注意前几个特殊值的判定==

```cpp
    int n = 500; //范围 
    int m = sqrt(n)+0.5;
    for(int i = 2; i<=m; i++) if(!vis[i])
        for(int j = i*i; j<=n; j+=i) vis[j] = 1;
```

### Euler筛法

==注意前几个特殊值的判定==

```cpp
#define N 10000
bool isPrime[N+5];
vector<int> prime;
void init() {
    memset(isPrime, true, sizeof(isPrime));
    for(int i = 2; i<=N; i++){
        if(isPrime[i]) prime.push_back(i);
        for(int j = 0; j < prime.size(); j++){
            if(i* prime[j] > N) break;
            isPrime[i*prime[j]] = false;
            if(i % prime[j] == 0) break;
        }
    }
    isPrime[0] = isPrime[1] = false;
}
   
```

防止卡STL版本

```cpp
const int N = 1e6+5;
bool isprime[N];
int p[N];
int top=0;
void init() {
    memset(isprime,1,sizeof(isprime));
    isprime[0]=isprime[1]=0;
    for(int i=2;i<N;i++) {
        if(isprime[i])
            p[top++]=i;
        for(int j=0;j<top&&(i*p[j])<N;j++) {
            isprime[i*p[j]]=0;
            if(i%p[j]==0) break;
        }
    }
}
```

### 单独判断

```cpp
bool prime(int x) {
	int i;
	for (i = 2; i <= int(sqrt(x)+0.5); i++) 
		if (x%i == 0) 
			return false;
	return true;
}
```

## 多元一次线性方程组



## 欧拉函数

在数论对正整数n，欧拉函数是小于n的正整数中与n互质的数的数目、

```cpp
const int N = 5e6+5;
int phi[N],vis[N],prime[N];
void init() {
    int cnt = 0;
    for (int i = 2; i < N; i++) {
        if (!vis[i])
            prime[cnt++] = i,phi[i] = i - 1;
        for (int j = 0; j < cnt && prime[j] < N / i; j++) {
            vis[prime[j] * i] = true;
            if (i % prime[j])
                phi[i * prime[j]] = phi[i] * (prime[j] - 1);
            else {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
        }
    }
}
```



单独判断

```cpp
int Euler2(int n)
{
    int res = n;
    for(int i = 2; i*i <= n; i++){
        if(n%i == 0) res = res/i*(i-1);
        while(n%i == 0) n/=i;
    }
    if(n > 1) res = res/n*(n-1);
    return res;
}

```





## 盛金公式（求解一元三次方程组）

![image-20210527214724558](C:\Users\FengLing\AppData\Roaming\Typora\typora-user-images\image-20210527214724558.png)

## 快速幂

### 递归

```cpp
int pow_mod(int a, int n, int p){
    a%=p;
    if(n==0)return 1;
    int x  = pow_mod(a, n/2, p);
    int ans = (int) x * x % p;
    if(n%2==1) ans = ans * a % p;
    return (int) ans;
}
```

### 无模

```cpp
int quickPower(int a, int b)
{
	int ans = 1, base = a;
	while(b > 0)
    {
		if(b & 1)
			ans *= base;
		
        base *= base;
		b >>= 1;
	}
	return ans;
}

```

### 有模

```cpp
int fastPow(int base, int power, int p) {
    base %= p;
    int ans = 1;
    while (power > 0) {
        if (power & 1) {
            ans = ans * base % p;
        }
        power /= 2;
        base = base * base % p;
    }
    return ans;
}
```

### 有模压行

```cpp
int pow(int a,int b){ 
	int res=1;
	a%=p;
	for(;b;b>>=1,a=a*a%p) if(b&1) res=res*a%p;
    return res;
}
```

## 组合数

### 杨辉三角

$0\le m \le n \le 1000$  , $1 \le p \le 1e9$ 时使用

```cpp
int dp[250][250];
const int p = 1000000007;

signed main() {

//#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
//#endif

    int a, b; cin >> a >> b;
    for(int i = 0; i<220; i++){
        for(int j = 0; j<=i; j++){
            if(j == 0 || i == j) dp[i][j] = 1;
            else dp[i][j] = (dp[i-1][j-1] + dp[i-1][j])%p;
        }
    }
    cout << dp[b][a];
    
    return 0;
}
```

### 阶乘逆元

可以用线性复杂度预处理进行优化。
同余定理+逆元: [https://blog.csdn.net/LOOKQAQ/article/details/81282342](https://blog.csdn.net/LOOKQAQ/article/details/81282342)

```cpp
int fact[250];
const int mod = 1000000007;

int fastPow(int base, int power, int p) {
    int ans = 1;
    while (power > 0) {
        if (power & 1) {
            ans = ans * base % p;
        }
        power >>= 1;
        base = base * base % p;
    }
    return ans;
}

int inv(int a, int p){
    return fastPow(a, p-2, mod);
}


signed main() {

//#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
//#endif

    int m, n; cin >> m >> n;

    fact[0] = 1;
    for(int i = 1; i<=220; i++){
        fact[i] = i * fact[i-1] % mod;
    }


    cout << fact[n] * inv(fact[m], mod) % mod * inv(fact[n-m], mod) % mod;


    return 0;
}
```

### 卢卡斯定理Lucas

原理：
$Lucas(n ,m, p)\ mod\ p\ = Lucas(\frac{n}{p}, \frac{m}{p}, p) * C^{m\ mod\ p}_{n\ mod\ p}\ mod\ p$
$Lucas(n ,m, p)\ mod\ p = C^m_n\ mod\ p$



一定要开long long，不然无法计算fact数组！！！

```cpp
int fact[250];
const int mod = 1000000007;

int fastPow(int a, int b, int p)
{
    int ans = 1;
    a %= p;
    while(b)
    {
        if(b & 1)
        {
            ans = ans * a % p;
            b--;
        }
        b >>= 1;
        a = a * a % p;
    }
    return ans;
}

int C(int n, int m, int p) {
    if (m > n) return 0;
    return (((fact[n] * fastPow(fact[m], p - 2, p)) % p) * fastPow(fact[n - m], p - 2, p)) % p;
}

int lucas(int n, int m, int p) {
    if (!m) return 1;
    return C(n % p, m % p, p) * lucas(n / p, m / p, p) % p;
}


signed main() {

#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    int m, n;
    cin >> m >> n;

    fact[0] = 1;
    for (int i = 1; i <= 220; i++) {
        fact[i] = i * fact[i - 1] % mod;
    }

    cout << lucas(n, m, mod);
    
    return 0;
}
```

## 卡特兰数

通项: $C_n=\frac{C_{2n}^{n}}{n+1}$

递推式: $C_1=1, C_n=C_{n-1} \frac{4*n-2}{n+1}$



Python打表

```python
ans, n = 1, 20
print("1:" + str(ans))
for i in range(2, n + 1):
    ans = ans * (4 * i - 2) // (i + 1)
    print(str(i) + ":" + str(ans))
```



Java打表

```java
import java.math.BigInteger;
public class Main {
    public static void main(String[] args) {
        // 打印前 n 个卡特兰数
        int n = 20;
        BigInteger ans = BigInteger.valueOf(1);
        System.out.println("1:" + ans.toString());
        BigInteger four = BigInteger.valueOf(4);
        BigInteger one = BigInteger.valueOf(1);
        BigInteger two = BigInteger.valueOf(2);
        for (int i = 2; i <= n; i++) {
            BigInteger bi = BigInteger.valueOf(i);
            ans = ans.multiply(four.multiply(bi).subtract(two)).divide(bi.add(one));
            System.out.println(i + ":" + ans.toString());
        }
    }
}


```



## 球盒问题

N球放M盒，其实有8种情况：

1. 球同，盒同，盒不可以为空

2. 球同，盒同，盒可以为空

3. 球同，盒不同，盒不可以为空

4. 球同，盒不同，盒可以为空

5. 球不同，盒同，盒不可以为空

6. 球不同，盒同，盒可以为空

7. 球不同，盒不同，盒不可以为空

8. 球不同，盒不同，盒可以为空



1 2 类情况，穷举法。

例如7个相同球放入4个相同盒子，每盒至少一个（1类情况），则先4个盒子每个放1个，多余3个。只需要考虑这3个球的去处就OK，由于盒子相同，所以只需要凑数就OK，不必考虑位置，因此只有300，211，111三种。

例如7个相同球放入4个相同盒子，可以空盒，则还是凑数，大的化小的，小的化更小的。

0007 0016 0025 0034 0115 0124 0133 0223 1114 1123 1222

11种。





3 4 类情况，用插板法（隔板法）解决。

3 的公式是把 N 个球排成一排（一种方法），它们中间有 N-1 个空。取 M-1 个板，放到空上，就把它们分成 M 部分，由于板不相邻，所以没有空盒。它的方法数有C(N-1, M-1)

4 的公式在3的基础上升华出来的，为了避免空盒，先在每一个盒里假装放一个球，这样就有 N+M 个球，C(N+M-1, M-1)



球不同的情况里，先来分析最特殊的8号：N球不同，M盒不同，允许空。每个球都有M种选择，N个球就有 M^N 种分法。



关于5 6 7 的情况，”我先教大家一个非常特殊的三角形，这个你在狗哥百度非常难以找的到的，秘传型，一般人我不会告诉他的。” (啊啊，可爱的原作者 —azalea注）

看起来很复杂，其实很简单：



![img](http://azaleasays.com/images/2011/12/triangle.jpg)





性质1，左右两边都是1，第几行就有几个数，比如第5行就是1XXX1

性质2， S(N, K) = S(N-1, K-1) + K * S(N-1, K)，含义是第N排的第K个数等于他上一排的上一个位置数字加上一排的同样位置数字的K倍。

例如S(7, 3) 就是第7排第3个数字，所以他等于上排第6排第2个数字+第6排第3个位置*3。所以画图的话，明显第1排是1，第2排1，1，推理第3排（左右两边都是1，只有中间那个数字没确定）。 所以 S(3, 2) = 第2排第1个数字+第2排第2个数字两倍 = 1+1*2 = 3，所以第3排数字就是1，3，1。同理 S(4, 2) = S(3, 1) + 2*S(3, 2) = 1+2*3 = 7, … 如此类推。

当遇见类型5即：N不同球，M同盒，无空盒。一共有 S(N, M) 种分法，比如7个不同球，4个相同箱子，每个箱子至少一个，则看三角形的第7行，第4个数字多少。

而类型6，N不同球，M同箱，允许空的时候（在类型5的基础上允许空箱）。明显是N个球不变，一个空箱子都没有+有一个空箱子+有两个空箱子+有三个空箱子+，，，，，，都装在一个箱子。说的简单点一共有就是

S(N, 1) + S(N, 2) + S(N, 3) + … + S(N, M)

也就是说第N排开始第1个数字一直加到第M个数字就是总的分法。

而类型7同样是在类型5的基础上升华，因为5是箱同的，而7箱不同，所以箱子自身多了P(M, M) = M! 倍可能， 所以类型7的公式就是 M! * S(N, M)

总结：

N球M盒

1. 球同，盒同，盒不可以为空 穷举

2. 球同，盒同，盒可以为空 穷举

3. 球同，盒不同，盒不可以为空 C(N-1, M-1

4. 球同，盒不同，盒可以为空 C(N+M-1, M-1)

5. 球不同，盒同，盒不可以为空 S(N, M)

6. 球不同，盒同，盒可以为空 S(N, 1) + S(N, 2) + S(N, 3) + … + S(N, M)

7. 球不同，盒不同，盒不可以为空 M! * S(N, M)

8. 球不同，盒不同，盒可以为空 M^N



# 计算几何



## AIZU板子

```cpp
#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <deque>
#include <algorithm>
#include <stack>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <list>
#include <bitset>
#include <complex>
#include <iomanip>
#include <assert.h>
/*-----------------------*/
#define int long long
#define INF 0x3f3f3f3f
#define INF2 0x7f7f7f7f
using namespace std;

/*-----------------------*/



using namespace std;

using int64 = long long;
//const int mod = 1e9 + 7;
const int mod = 998244353;

const int64 infll = (1LL << 62) - 1;
const int inf = (1 << 30) - 1;

struct IoSetup {
    IoSetup() {
        cin.tie(nullptr);
        ios::sync_with_stdio(false);
        cout << fixed << setprecision(10);
        cerr << fixed << setprecision(10);
    }
} iosetup;


template< typename T1, typename T2 >
ostream &operator<<(ostream &os, const pair< T1, T2 > &p) {
    os << p.first << " " << p.second;
    return os;
}

template< typename T1, typename T2 >
istream &operator>>(istream &is, pair< T1, T2 > &p) {
    is >> p.first >> p.second;
    return is;
}

template< typename T >
ostream &operator<<(ostream &os, const vector< T > &v) {
    for(int i = 0; i < (int) v.size(); i++) {
        os << v[i] << (i + 1 != v.size() ? " " : "");
    }
    return os;
}

template< typename T >
istream &operator>>(istream &is, vector< T > &v) {
    for(T &in : v) is >> in;
    return is;
}

template< typename T1, typename T2 >
inline bool chmax(T1 &a, T2 b) { return a < b && (a = b, true); }

template< typename T1, typename T2 >
inline bool chmin(T1 &a, T2 b) { return a > b && (a = b, true); }

template< typename T = int64 >
vector< T > make_v(size_t a) {
    return vector< T >(a);
}

template< typename T, typename... Ts >
auto make_v(size_t a, Ts... ts) {
    return vector< decltype(make_v< T >(ts...)) >(a, make_v< T >(ts...));
}

template< typename T, typename V >
typename enable_if< is_class< T >::value == 0 >::type fill_v(T &t, const V &v) {
    t = v;
}

template< typename T, typename V >
typename enable_if< is_class< T >::value != 0 >::type fill_v(T &t, const V &v) {
    for(auto &e : t) fill_v(e, v);
}

template< typename F >
struct FixPoint : F {
    FixPoint(F &&f) : F(forward< F >(f)) {}

    template< typename... Args >
    decltype(auto) operator()(Args &&... args) const {
        return F::operator()(*this, forward< Args >(args)...);
    }
};

template< typename F >
inline decltype(auto) MFP(F &&f) {
    return FixPoint< F >{forward< F >(f)};
}

namespace geometry {
    using Real = double;
    const Real EPS = 1e-8;
    const Real PI = acos(static_cast< Real >(-1));

    inline int sign(const Real &r) {
        return r <= -EPS ? -1 : r >= EPS ? 1 : 0;
    }

    inline bool equals(const Real &a, const Real &b) {
        return sign(a - b) == 0;
    }
}

namespace geometry {
    using Point = complex< Real >;

    istream &operator>>(istream &is, Point &p) {
        Real a, b;
        is >> a >> b;
        p = Point(a, b);
        return is;
    }

    ostream &operator<<(ostream &os, const Point &p) {
        return os << real(p) << " " << imag(p);
    }

    Point operator*(const Point &p, const Real &d) {
        return Point(real(p) * d, imag(p) * d);
    }

    // rotate point p counterclockwise by theta rad
    Point rotate(Real theta, const Point &p) {
        return Point(cos(theta) * real(p) - sin(theta) * imag(p), sin(theta) * real(p) + cos(theta) * imag(p));
    }

    Real cross(const Point &a, const Point &b) {
        return real(a) * imag(b) - imag(a) * real(b);
    }

    Real dot(const Point &a, const Point &b) {
        return real(a) * real(b) + imag(a) * imag(b);
    }

    bool compare_x(const Point &a, const Point &b) {
        return equals(real(a), real(b)) ? imag(a) < imag(b) : real(a) < real(b);
    }

    bool compare_y(const Point &a, const Point &b) {
        return equals(imag(a), imag(b)) ? real(a) < real(b) : imag(a) < imag(b);
    }

    using Points = vector< Point >;
}

namespace geometry {
    struct Line {
        Point a, b;

        Line() = default;

        Line(const Point &a, const Point &b) : a(a), b(b) {}

        Line(const Real &A, const Real &B, const Real &C) { // Ax+By=C
            if(equals(A, 0)) {
                assert(!equals(B, 0));
                a = Point(0, C / B);
                b = Point(1, C / B);
            } else if(equals(B, 0)) {
                a = Point(C / A, 0);
                b = Point(C / A, 1);
            } else {
                a = Point(0, C / B);
                b = Point(C / A, 0);
            }
        }

        friend ostream &operator<<(ostream &os, Line &l) {
            return os << l.a << " to " << l.b;
        }

        friend istream &operator>>(istream &is, Line &l) {
            return is >> l.a >> l.b;
        }
    };

    using Lines = vector< Line >;
}

namespace geometry {
    struct Segment : Line {
        Segment() = default;

        using Line::Line;
    };

    using Segments = vector< Segment >;
}

namespace geometry {
    constexpr int COUNTER_CLOCKWISE = +1;
    constexpr int CLOCKWISE = -1;
    constexpr int ONLINE_BACK = +2; // c-a-b
    constexpr int ONLINE_FRONT = -2; // a-b-c
    constexpr int ON_SEGMENT = 0; // a-c-b
    int ccw(const Point &a, Point b, Point c) {
        b = b - a, c = c - a;
        if(sign(cross(b, c)) == +1) return COUNTER_CLOCKWISE;
        if(sign(cross(b, c)) == -1) return CLOCKWISE;
        if(sign(dot(b, c)) == -1) return ONLINE_BACK;
        if(norm(b) < norm(c)) return ONLINE_FRONT;
        return ON_SEGMENT;
    }
}

namespace geometry {
    Point projection(const Line &l, const Point &p) {
        auto t = dot(p - l.a, l.a - l.b) / norm(l.a - l.b);
        return l.a + (l.a - l.b) * t;
    }
}

namespace geometry {
    Real distance_lp(const Line &l, const Point &p) {
        return abs(p - projection(l, p));
    }
}

namespace geometry {
    bool is_parallel(const Line &a, const Line &b) {
        return equals(cross(a.b - a.a, b.b - b.a), 0.0);
    }
}

namespace geometry {
    bool is_intersect_ll(const Line &l, const Line &m) {
        Real A = cross(l.b - l.a, m.b - m.a);
        Real B = cross(l.b - l.a, l.b - m.a);
        if(equals(abs(A), 0) && equals(abs(B), 0)) return true;
        return !is_parallel(l, m);
    }
}

namespace geometry {
    bool is_intersect_sp(const Segment &s, const Point &p) {
        return ccw(s.a, s.b, p) == ON_SEGMENT;
    }
}


namespace geometry {
    Real distance_ll(const Line &l, const Line &m) {
        return is_intersect_ll(l, m) ? 0 : distance_lp(l, m.a);
    }
}

namespace geometry {
    Real distance_sp(const Segment &s, const Point &p) {
        Point r = projection(s, p);
        if(is_intersect_sp(s, r)) return abs(r - p);
        return min(abs(s.a - p), abs(s.b - p));
    }
}

namespace geometry {
    bool is_intersect_ss(const Segment &s, const Segment &t) {
        return ccw(s.a, s.b, t.a) * ccw(s.a, s.b, t.b) <= 0 &&
               ccw(t.a, t.b, s.a) * ccw(t.a, t.b, s.b) <= 0;
    }
}


namespace geometry {
    Real distance_ss(const Segment &a, const Segment &b) {
        if(is_intersect_ss(a, b)) return 0;
        return min({distance_sp(a, b.a), distance_sp(a, b.b), distance_sp(b, a.a), distance_sp(b, a.b)});
    }
}

using namespace geometry;



signed main() {

#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif



    return 0;


}
```



## 点和向量

```cpp
struct Point{
    double x, y;
    Point(double x=0, double y=0) :x(x), y(y){ }
};

typedef Point Vector;

Vector operator + (Vector A, Vector B){ return Vector(A.x+B.x, A.y+B.y);}

Vector operator - (Point A, Point B) { return Vector(A.x-B.x, A.y-B.y);}

Vector operator * (Vector A, double p) { return Vector(A.x*p, A.y*p);}

Vector operator / (Vector A, double p) { return Vector(A.x/p, A.y/p);}

bool operator < (const Point& a, const Point& b){
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

const double eps = 1e-10;
int dcmp(double x){
    if(fabs(x) < eps) return 0; else return x < 0 ? -1 : 1;
}

bool operator == (const Point& a, const Point& b){
    return dcmp(a.x-b.x) == 0 && dcmp(a.y-b.y) == 0;
}

double Dot(Vector A, Vector B) {return A.x*B.x + A.y*B.y;}
double Length(Vector A) { return sqrt(Dot(A, A));}
double Angle(Vector A, Vector B) { return acos(Dot(A, B) / Length(A) / Length(B));}

double Cross(Vector A, Vector B) { return A.x*B.y - A.y*B.x;}
double Area2(Point A, Point B, Point C) { return Cross(B-A, C-A);}

Vector Rotate(Vector A, double rad){
    return Vector(A.x*cos(rad)-A.y*sin(rad), A.x*sin(rad)+A.y*cos(rad));
}

//计算向量的单位法线。即左转90度，再把长度归一化。
Vector Normal(Vector A){
    double L = Length(A);
    return Vector(-A.y/L, A.x/L);
}

//点到直线的投影
Point DropFeet(Point P, Point A, Point B){
    Vector AB = B-A;
    Vector AP = P-A;
    double AD_len = Dot(AP, AB) / Length(AB);
    Vector AD = AB / Length(AB) * AD_len;
    Point D = A + AD;
    return D;
}

//对称点
Point SymmetryPoint(Point P, Point A, Point B){
    Vector AB = B-A;
    Vector AP = P-A;
    double AD_len = Dot(AP, AB) / Length(AB);
    Vector AD = AB / Length(AB) * AD_len;
    Point D = A + AD;
    Vector PD = D-P;
    Vector PQ = PD * 2;
    Point Q  = PQ + P;
    return Q;
}

//直线AB与直线CD的交点（我也不知道平行会发生什么）
Point Intersection(Point A, Point B, Point C, Point D){
    Vector AB = B-A;
    Vector AP =  AB * (abs(Area2(A, C, D))/(abs(Area2(A, C, D)) + abs(Area2(B, C, D))));
    Point P = A + AP;
    return P;
}

//点是否在线段上
bool isOn(Point P, Point A, Point B){
    Vector AB = B-A;
    Vector AP = P-A;
    Vector PB = B-P;
    return dcmp(Length(AP) + Length(PB) - Length(AB))==0;
}

```









## 两圆位置关系

两元外切 $d=R+r$

两圆外离 $d>R+r$

两圆内含 $d<R-r$

两圆相交 $R-r<d<R+r$

两圆内切 $d=R-r$



## 两圆交点

题目：

　　给定两个圆的的方程

　　　　(x-x1)^2 +(y-y1)^2 =r1^2,

　　　　(x-x2)^2 +(y-y2)^2 =r2^2

　　求解两个圆的交点坐标。

这种知识是高中的知识了，如果直接联立，由于计算特别暴力所以很难得到正确结果。

事实上，造成这种计算问题的结果是因为消去一个变量后，剩下变量的系数太过复杂。

我们通过更改坐标系的方法来使得另外一个系数变得简单。

我们先计算中点坐标：![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220207397-117438393.png)

圆心距：![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220222912-1830306010.png)

 

设定两个新的正交单位向量作为新的坐标系的x和y轴，而原点为圆心线段的中点, 设为M：

![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220309147-1864053348.png)

在新的坐标系中, 圆1的方程变为：

　　　　　　　　　　![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220422053-468300829.png) （1）

圆2的方程变为：

　　　　　　　　　　![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220447022-1326058629.png)（2）

联立（1）（2）两式，得到a=![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220619272-711819577.png)

　　![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220631444-995544904.png)

最后的交点变换回原来的坐标![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123221345162-917282671.png)： 

![img](https://images2017.cnblogs.com/blog/891812/201801/891812-20180123220655022-606748754.png)



```cpp
pair<Point, Point> inter(Circle a, Circle b){

    double R = sqrt(squ(a.x-b.x) + squ(a.y-b.y));
    Point ans1 = Point(a.x+b.x, a.y+b.y)*0.5 + Point(b.x-a.x, b.y-a.y)*(squ(a.r) - squ(b.r))/(2*squ(R))
                + Point(b.y-a.y, a.x - b.x) *0.5 * sqrt(2*((squ(a.r)+squ(b.r))/squ(R)) - squ(squ(a.r) - squ(b.r))/squ(squ(R)) - 1);
    Point ans2 = Point(a.x+b.x, a.y+b.y)*0.5 + Point(b.x-a.x, b.y-a.y)*(squ(a.r) - squ(b.r))/(2*squ(R))
                 - Point(b.y-a.y, a.x - b.x) *0.5 * sqrt(2*((squ(a.r)+squ(b.r))/squ(R)) - squ(squ(a.r) - squ(b.r))/squ(squ(R)) - 1);
    return make_pair(ans1, ans2);
}
```







## 三角形面积

海伦公式 $S=\sqrt{p(p-a)(p-b)(p-c)}$     其中 $a, b, c$ 为三角形的三边长， $p$为半周长







# 图论



## 拓扑排序

​		把每个变量看成一个点，“小于”关系看成有向边，则得到了一个有向图。这样，我们的任务实际上是把一个图的所有结点排序，使得每一条有向边$(u,v)$对应的$u$都排在$v$的前面。在图论中，这个问题成为拓扑排序(topological sort)

​		不难发现：如果图中存在有向环，则不存在拓扑排序，反之则存在。不包含有向环的有向图称为有向无环图(Directed Acyclic Graph, DAG)。可以借助DFS完成拓扑排序：在访问一个结点之后把它加到当前拓扑排序的首部（想一想，为什么不是尾部）。

​		这里用到了一个c数组，c[u]=0表示从来没有访问过（从来没有调用过dfs(u)）；c[u]=1表示已经访问过，并且还递归访问过它的所有子孙（即dfs(u)曾被调用过，并以返回）；c[u]=-1表示正在访问（即递归调用dfs(u)正在栈帧中，尚未返回）。

### 仅判断是否可排

```cpp
bool dfs(int u){
    c[u] = -1;
    for(int v = 0; v<n; v++) if(g[u][v]){
        if(c[v] < 0) return false;
        else if(!c[v] && !dfs(v)) return false;
    }
    c[u] = 1;
    return true;
}

bool toposort(){
    memset(c, 0, sizeof(c));
    for(int u = 0; u<n; u++) if(!c[u])
        if(!dfs(u)) return false;
    return true;
}
```

### 完整代码

```cpp
#define maxn 105
int c[maxn];
int topo[maxn], t;
int g[maxn][maxn];

int n, m;

bool dfs(int u){
    c[u] = -1;
    for(int v = 0; v<n; v++) if(g[u][v]){
        if(c[v] < 0) return false;
        else if(!c[v] && !dfs(v)) return false;
    }
    c[u] = 1; topo[--t]=u;
    return true;
}

bool toposort(){
    t = n;
    memset(c, 0, sizeof(c));
    for(int u = 0; u<n; u++) if(!c[u])
        if(!dfs(u)) return false;
    return true;
}
```





## 字符串树

```cpp
const int maxn = 1000;
int lch[maxn], rch[maxn]; char op[maxn]; //每个结点的左右子节点编号和字符
int nc = 0; //结点数

int build_tree(char* s, int x, int y){
    int i, c1 = -1, c2 = -1, p = 0;
    int u;
    if(y-x == 1){ //边界情况: 仅一个字符，建立单独节点
        u = ++nc;
        lch[u] = rch[u] = 0; op[u] = s[x];
        return u;
    }

    for(i = x; i<y; i++){
        switch(s[i]){
            case '(': p++; break;
            case ')': p--; break;
            case '+': case '-': if(!p) c1 = i; break;
            case '*': case '/': if(!p) c2 = i; break;
        }
    }

    if(c1 < 0) c1 = c2; //找不到加减就用乘除
    if(c1 < 0) return build_tree(s, x+1, y-1); //整个表达式被一对括号括起来的情况
    u = ++nc;
    lch[u] = build_tree(s, x, c1);
    rch[u] = build_tree(s, c1+1, y);
    op[u] = s[c1];
    return u;

}
```



## 欧拉回路





## 迪杰斯特拉（Dijkstra）

==注意每次要将未连通的边```g[i][j]```设置为INF==

N		图中的点数

g		图 ```g[i][j]```表示结点i与j的距离

dis	```dis[i]```表示结点0到i的最短距离

```cpp
const int maxn = 105;
int dis[maxn];
int g[maxn][maxn];
int N;
bool v[maxn];

void dijkstra(){
    for(int i = 0; i<N; i++) dis[i] = INF;
    dis[0] = 0;
    memset(v, 0, sizeof(v));
    for(int i = 0; i<N; i++){
        int mark = -1, mindis = INF;
        for(int j = 0; j<N; j++){
            if(!v[j] && dis[j]<mindis){
                mindis=dis[j];
                mark = j;
            }
        }
        v[mark] = 1;
        for(int j = 0; j<N; j++) if(!v[j])
                dis[j]=min(dis[j], dis[mark]+g[mark][j]);
    }

}
```



## 弗洛伊德（Floyd-Warshall）

​		求任意两个点之间的最短路径。这个问题这也被称为“多源最短路径”问题。Floyd-Warshall算法不能解决带有“负权回路”（或者叫“负权环”）的图，因为带有“负权回路”的图没有最短路。

### 求最短路

要将不连通的路初始化为INF

```cpp
for (k = 1; k <= n; k++)
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (e[i][j] > e[i][k] + e[k][j])
                e[i][j] = e[i][k] + e[k][j];
```

### 求传递闭包

```cpp
for(int k = 1; k<=n; k++)
            for(int i = 1; i<=n; i++)
                for(int j = 1; j<=n; j++)
                    if(g[i][k] && g[k][j]) g[i][j] = 1;

        int ans = 0;
        for(int i = 1; i<=n; i++){
            int flag = 1;
            for(int j = 1; j<=n; j++){
                if(i!=j && !g[i][j] && !g[j][i])
                    flag = 0;
            }
            ans += flag;
        }
```



## 最小生成树（Kruskal）

```cpp
const int maxn = 1e5+500;
 
int u[maxn];
int v[maxn];
int w[maxn];
 
int p[maxn];
int r[maxn];
 
int cmp (const int i, const int j){
    return w[i] < w[j];
}
 
int find(int x){
    return p[x] == x ? x : p[x] = find(p[x]);
}
 
int Kruskal(){
    int ans = 0;
    for(int i = 0; i < maxn; i++) p[i] = i;
    for(int i = 0; i < maxn; i++) r[i] = i;
    sort(r, r + maxn, cmp);
    for(int i = 0; i < maxn; i++){
        int e = r[i]; int x = find(u[e]); int y = find(v[e]);
        if(x!=y){ans += w[e]; p[x] = y;}
    }
    return ans;
}
```



## LCA 最近公共祖先

是指在有根树中，找出某两个结点u和v最近的公共祖先。



### 倍增

```cpp
const int maxn = 500000 + 50;
vector<int> g[maxn];
int grd[maxn][23];
int dep[maxn];

void dfs(int u, int fa){
    grd[u][0] = fa;
    dep[u] = dep[fa]+1;
    for(int i = 1; i<21; i++)
        grd[u][i] = grd[grd[u][i-1]][i-1];
    int sz = g[u].size();
    for(int i = 0; i<sz; i++){
        if(g[u][i] == fa) continue;
        dfs(g[u][i], u);
    }
}

int lca(int x, int y){
    if(dep[x] < dep[y]) swap(x, y);
    for(int i = 20; i>=0; i--){
        if((1<<i) <= dep[x] - dep[y])
            x = grd[x][i];
    }
    if(x == y) return x;
    for(int i = 20; i>=0; i--){
        if(grd[x][i] != grd[y][i])
            x = grd[x][i], y = grd[y][i];
    }
    return grd[x][0];
}
```

### Tarjan





# 数据结构



## KMP

```cpp
void getFail(char* P, int* f){
    int m = strlen(P);
    f[0] = f[1] = 0;
    for(int i = 1; i<m; i++){
        int j = f[i];
        while(j && P[i]!=P[j]) j = f[j];
        f[i+1] = P[i] == P[j] ? j+1 : 0;
    }
}

void find(char* T, char* P, int* f){
    int n = strlen(T), m = strlen(P);
    getFail(P, f);
    int j = 0;
    for(int i = 0; i<n; i++){
        while(j && P[j]!=T[i]) j = f[j];
        if(P[j] == T[i]) j++;
        if(j == m) printf("%d\n", i-m+1); //find successfully
    }
}
```



## Trie前缀树 字典树

要注意将前缀树开在主函数外来避免堆栈溢出。

### 字符串

```cpp
#define maxnode 50500
#define sigma_size 26

struct Trie {
    int ch[maxnode][sigma_size];
    int val[maxnode];
    int sz;

    Trie() {
        sz = 1;
        memset(ch[0], 0, sizeof(ch[0]));
    }

    int idx(char c) { return c - 'a'; }

    void insert(string s, int v) {
        int u = 0, n = s.length();
        for (int i = 0; i < n; i++) {
            int c = idx(s[i]);
            if (!ch[u][c]) {
                memset(ch[sz], 0, sizeof(ch[ssz]));
                val[sz] = 0;
                ch[u][c] = sz++;
            }
            u = ch[u][c];
        }
        val[u] = v;
    }

    int find(string s){
        int len = s.length();
        int u = 0;
        for(int i = 0; i<len; i++)
        {
            int c = idx(s[i]);
            if(!ch[u][c]) return 0;
            u = ch[u][c];
        }
        return val[u];
    }
};
```

### 字符数组

```cpp
//sigma_size=26 maxnode为节点最大数：字符串数*每个字符串最大长度+x
struct trie
{
	int ch[maxnode][sigma_size];//ch[a][3]=c表示第a个节点有一个子节点为'd'并且节点序列号为c
	int val[maxnode];//存各个字符串权值，题目不涉及权值时，可以存1
	int sz;
	int idx(char c) { return c - 'a';}//找到char c对应的id值
	void init()
	{
		sz = 1;
		memset(ch[0], 0, sizeof(ch[0]));
	}
	void insert(char *s, int v)//s为字符串，v为权值
	{
		int u = 0, n = strlen(s);
		for(int i = 0; i < n; i++)
		{
			int c = idx(s[i]);
			if(!ch[u][c])
			{
				memset(ch[sz], 0, sizeof(ch[sz]));
				val[sz] = 0;
				ch[u][c] = sz++;
			}
			u = ch[u][c];
		}
		val[u] = v;
	}
	int find(char *s)//查找s，返回权值
	{
		int u = 0;
		for(int i = 0; s[i]; i++)
		{
			int c = idx(s[i]);
			if(!ch[u][c]) return 0;
			u = ch[u][c];
		}
		return val[u];
	}
}Trie;
```

## 并查集

### 类解法

```cpp
//并查集实现
class UnionFindSet{
    vector<int> F;//并查集容器
    vector<int> rank;//秩优化(如果两个都有很多元素的根节点相遇，将根节点选为元素较少的那一个，可以节省时间)
    int n;
 
public:
    //并查集初始化
    UnionFindSet(int _n){
        n = _n;
        F.resize(n);
        rank.resize(n, 1);
        for(int i = 0; i < n; i++){
            F[i] = i;
        }
    }
 
    //并查集查询操作
    int find(int x){
        return x == F[x] ? x : F[x] = find(F[x]);//查询优化。找到x的根节点
    }
 
    //并查集合并操作
    bool merge(int x, int y){
        int fx = find(x), fy = find(y);
        if(fx == fy) return false;//两个节点连在同一个根节点上则直接返回 不再合并
 
    //因为是将节点数少的连接到节点数多的节点上，当fx下面的节点数小于fy下面的节点数时，交换fx和fy
    //即两个根节点相遇时，将新的根节点选为节点数较多的那一个，尽量减少find(x)的次数
        if(rank[fx] < rank[fy])//合并优化
            swap(fx, fy);
 
        F[fy] = fx;//将fy连到fx上(将节点数少的连接到节点数多的上面)
        rank[fx] += rank[fy];//将fy连到fx后 fx下面的节点数目要更新
        return true;
    }
};
```

### 函数解法

```cpp
#include <iostream>
#include<vector>
#include<string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>//算法头文件
#include <numeric>
#include <stack>
#include<typeinfo>
using namespace std;
 
 
 
vector<int> F;
// 初始化
void UnionFindSet(int N)
{
	F.resize(N + 1, -1);//如果初始化是从1开始，这里就得用N+1 如果从0开始初始化 就用N（从多少开始初始化 根据输入的数据进行调整）
	for (int i = 1; i <= N; i++)
	{
		F[i] = i; // 初始化时， 将自己初始化为自己的领导
	}
}
 
// 查找
int find(int n)
{
	return n == F[n] ? n : find(F[n]);
}
 
// 合并 这里直接在主程序里调用了查找 能够省去调用合并函数的次数
void merge(int leaderX, int leaderY)
{
	if (leaderX < leaderY)
	{
		F[leaderX] = leaderY;
	}
	else
	{
		F[leaderY] = leaderX;
	}
}
 
// 输入数组, 每一行表示一个集合关系， 比如第一行表示3和4属于一个集合
int input[] =
{
 1, 4,
 2, 5,
 3, 6,
 4, 2,
 5, 1,
 6, 3,
};
 
int main()
{
	int N;
	cin >> N;
	int numberOfSets = N;
	// 初始化领导
	UnionFindSet(N);
 
	int n = sizeof(input) / sizeof(input[0]) / 2;//这地方根据输入的形式调整
	int j = 0;
	for (int i = 0; i < n; i++)
	{
		int u = input[j++];
		int v = input[j++];
		u = find(u);//这里直接在主程序里调用了查找 能够省去调用合并函数的次数
		v = find(v);
		if (u != v) {	//如果没关系 就合并 最后numberOfSets就是不能合并的个数 也就是有几个圈子
			merge(u, v);
			numberOfSets--;
		}
	}
	cout << numberOfSets << endl;
 
	return 0;
}
```



## 树状数组（二叉索引树 BIT）

适用于单点修改、区间求和查询

### 函数解法

注意这个n时树状数组的n，不要和其它的n混淆。

```cpp
int c[1000020];
int n;

int lowbit(int x){
    return x & -x;
}

int sum(int x){
    int ret = 0;
    while(x > 0){
        ret += c[x]; x-= lowbit(x);
    }
    return ret;
}

void add(int x, int d){
    while(x <= n){
        c[x] += d; x += lowbit(x);
    }
}
```

### 结构体解法

```cpp
struct BIT
{
    int n;
    vector<int>C;
    void resize(int n){ this->n=n;C.resize(n);}
    void clear(){ fill(C.begin(),C.end(),0);}
    int lowbit(int x){ return x&(-x);}
    int sum(int x){
        int ret=0;
        while(x>0){
            ret+=C[x];
            x-=lowbit(x);
        }
        return ret;
    }
    void add(int x,int add){
        while(x<=n)
        {
            C[x]+=add;
            x+=lowbit(x);
        }
    }
};
```

## RMQ 区间最小值问题 

Tarjan的Sparse-Table算法。$O(nlogn)$初始化，$O(1)$查询

此问题也有$O(n)$初始化的算法但过于复杂。

```cpp
int d[100000][20];

void RMQ_init(const vector<int> &A) {
    int n = A.size();
    for (int i = 0; i < n; i++) d[i][0] = A[i];
    for (int j = 1; (1 << j) <= n; j++)
        for (int i = 0; i + (1 << j) - 1 < n; i++)
            d[i][j] = min(d[i][j - 1], d[i + (1 << (j - 1))][j - 1]);
}

int RMQ(int l, int r) {
    int k = 0;
    while ((1 << (k + 1)) <= r - l + 1)k++;
    return min(d[l][k], d[r - (1 << k) + 1][k]);
}
```



## 线段树(1)

单点修改，区间查询

- $update(x,v)$：把$A_x$修改为$v$
- $query(l, r)$：计算$min(A_l,A_l+1,···,A_r)$

```cpp
const int maxn = 1000;
int num[maxn];
int minv[maxn*2];

int ql, qr; //查询[ql, qr]中的最小值
int query(int o, int L, int R){
    int M = (L + R) / 2, ans = INF;
    if(ql <= L && R <= qr) return minv[o];
    if(ql <= M) ans = min(ans, query(o*2, L, M));
    if(M < qr) ans = min(ans, query(o*2+1, M+1, R));
    return ans;
}

int p, v; //修改num[p] = v;
void update(int o, int L, int R){
    int M = (L + R) / 2;
    if(L == R) minv[o] = v;
    else{
        if(p <= M) update(o*2, L, M);
        else update(o*2+1, M+1, R);
        minv[o] = min(minv[o*2], minv[0*2+1]);
    }
}
```

## 线段树(2)

区间修改，区间查询

- add(L,R,v)：把$A_L,A_{L+1},···,A_R$的值全部增加v
- query(L,R)：计算子序列$A_L,A_{L+1},···,A_R$的元素和、最小值、最大值

```cpp
#define maxn 1000
int sumv[maxn];
int minv[maxn];
int maxv[maxn];
int addv[maxn];

void maintain(int o, int L, int R){
    int lc = o << 1; int rc = o << 1 | 1;
    sumv[o] = minv[o] = maxv[o] = 0 ;
    if(R > L){ //考虑左右子树
        sumv[o] = sumv[lc] + sumv[rc];
        minv[o] = min (minv[lc], minv[rc]);
        maxv[o] = max (maxv[lc], maxv[rc]);
    }
    minv[o] += addv[o]; maxv[o] += addv[o]; sumv[o] += (R - L + 1) * addv[o]; //考虑add操作
}

int a, b;
int v;
void update(int o, int L, int R){
    int lc = o << 1; int rc = o << 1 | 1;
    if(a <= L && R <= b){   //递归边界
        addv[o] += v;       //累加边界的add值
    }else{
        int M = (L + R) >> 1;
        if(a <= M ) update(lc, L, M);
        if(b > M) update(rc, M+1, R);
    }
    maintain(o, L, R);      //递归结束前重新计算本结点的附加信息
}

int _min, _max, _sum; //全局变量，目前位置的最小值、最大值和累加值
void query(int o, int L, int R, int add){
    if(a <= L && R <= b){ //递归边界：用边界区间的附加信息更新答案
        _sum += sumv[o] + add * (R-L+1);
        _min = min(_min, minv[o] + add);
        _max = max(_max, maxv[o] + add);
    }else{ //递归统计累加参数add
        int M = (L + R) >> 1;
        if(a <= M) query(o<<1, L, M, add+addv[o]);
        if(b > M) query(o<<1|1, M+1, R, add + addv[o]);
    }
}
```

## 线段树(3)



## Morris中序遍历

```cpp
TreeNode *cur = root, *pre = nullptr;
while (cur) {
    if (!cur->left) {
        // ...遍历 cur
        cur = cur->right;
        continue;
    }
    pre = cur->left;
    while (pre->right && pre->right != cur) {
        pre = pre->right;
    }
    if (!pre->right) {
        pre->right = cur;
        cur = cur->left;
    } else {
        pre->right = nullptr;
        // ...遍历 cur
        cur = cur->right;
    }
}
```





# 动态规划

## 背包问题

### 01背包

```cpp
void ZeroOnePack(int w, int v, int lim) {
    for (int i = lim; i >= w; i--)
        dp[i] = max(dp[i], dp[i - w] + v);
}
```

### 完全背包

```cpp
void CompletePack(int w, int v, int lim) {
    for (int i = w; i <= lim; i++)
        dp[i] = max(dp[i], dp[i - w] + v);
}
```

### 多重背包

```cpp
void ZeroOnePack(int w, int v, int lim) {
    for (int i = lim; i >= w; i--)
        dp[i] = max(dp[i], dp[i - w] + v);
}

void CompletePack(int w, int v, int lim) {
    for (int i = w; i <= lim; i++)
        dp[i] = max(dp[i], dp[i - w] + v);
}

void MultiplePack(int w, int v, int cnt, int lim) {
    if (cnt * w >= lim) {
        CompletePack(w, v, lim);
        return;
    } else {
        int k = 1;
        while (k <= cnt) {
            ZeroOnePack(k * w, k * v, lim);
            cnt -= k;
            k = k * 2;
        }
        ZeroOnePack(cnt * w, cnt * v, lim);
        return;
    }
}
```



# 实用技巧



## __int128

```cpp
void scan(__int128 &x)//输入
{
    x = 0;
    int f = 1;
    char ch;
    if((ch = getchar()) == '-') f = -f;
    else x = x*10 + ch-'0';
    while((ch = getchar()) >= '0' && ch <= '9')
    x = x*10 + ch-'0';
    x *= f;
}

void print(__int128 x)//输出
{
    if(x < 0){
        x = -x;
        putchar('-');
    }
    if(x > 9) print(x/10);
    putchar(x%10 + '0');
}
```





## onglu快读

```cpp
int read() {
    char c; int num, f = 1;
    while(c = getchar(),!isdigit(c)) if(c == '-') f = -1; num = c - '0';
    while(c = getchar(), isdigit(c)) num = num * 10 + c - '0';
    return f * num;
}
```



## 快读

```cpp
namespace fastIO {
    #define BUF_SIZE 100000
    //fread -> read
    bool IOerror = 0;
    inline char nc() {
        static char buf[BUF_SIZE], *p1 = buf + BUF_SIZE, *pend = buf + BUF_SIZE;
        if(p1 == pend) {
            p1 = buf;
            pend = buf + fread(buf, 1, BUF_SIZE, stdin);
            if(pend == p1) {
                IOerror = 1;
                return -1;
            }
        }
        return *p1++;
    }
    inline bool blank(char ch) {
        return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
    }
    inline void read(long long int &x) {
        char ch;
        while(blank(ch = nc()));
        if(IOerror)
            return;
        for(x = ch - '0'; (ch = nc()) >= '0' && ch <= '9'; x = x * 10 + ch - '0');
    }
    #undef BUF_SIZE
};
```



## 数组去重

要注意的是，先进行排序，再进行去重，原因有如下两点：

1. ```unique```只能实现在相同元素相邻的情况下进行去重。

2. 去重之后再进行排序会使得已经被去重的元素重新回到数组中。

```cpp
sort(a, a+n);
n = unique(a,a+n)-a;
```



直接去重

```cpp
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());
```



## 进制转换

题目来源：洛谷P1143
输入n、n进制的数、m 。输出：m进制对应的数
从数学思维上来讲，最好理解的方法是：先转换成10进制（根据位值原理），再取模运算，将10进制转换为m进制 。
样例：
输入：16 FF
输出：11111111

```cpp
string a;
int n,m,c[1000000],i,j=0,x,sum;
int main(){
    cin>>n;
    cin>>a;
    cin>>m;
    for(int i=0;i<a.size();i++){
        if(a[i]<'A'){   //0-9可以直接-'0'得出对应的数码值
            x=pow(n,a.size()-i-1);
            x*=(a[i]-'0');
            sum+=x;
        }
        else{    //A-F
            x=pow(n,a.size()-1-i);
            x*=(a[i]-'A'+10);
            sum+=x;
        }
    }
    while(sum>0){   //十进制转m进制 权值从小到大存余数
        c[j++]=sum%m;
        sum/=m;
    }
    for(i=j-1;i>=0;i--){  //模拟数学思想输出，注：此处不可用字符串函数来表示长度，因为c[]是整型数组
        if(c[i]<10) printf("%d",c[i]);
        else printf("%c",c[i]+'A'-10);
    }
    return 0;
}

```

## 高精度（大整数）

### 大整数取模

```cpp
int big_mod(string str, int p){
    int len = str.length();
    int res = 0;
    for(int i = 0; i<len; i++)
        res = (int) (((long long )res * 10 + str[i] - '0')%p);
    return res;
}
```

### 紫书模板（加、乘）

```cpp
#include<cstdio>
#include<cstring>
#include<vector>
#include<iostream>
using namespace std;

struct BigInteger {
  static const int BASE = 100000000;
  static const int WIDTH = 8;
  vector<int> s;

  BigInteger(long long num = 0) { *this = num; } // 构造函数
  BigInteger operator = (long long num) { // 赋值运算符
    s.clear();
    do {
      s.push_back(num % BASE);
      num /= BASE;
    } while(num > 0);
    return *this;
  }
  BigInteger operator = (const string& str) { // 赋值运算符
    s.clear();
    int x, len = (str.length() - 1) / WIDTH + 1;
    for(int i = 0; i < len; i++) {
      int end = str.length() - i*WIDTH;
      int start = max(0, end - WIDTH);
      sscanf(str.substr(start, end-start).c_str(), "%d", &x);
      s.push_back(x);
    }
    return *this;
  }
  BigInteger operator + (const BigInteger& b) const {
    BigInteger c;
    c.s.clear();
    for(int i = 0, g = 0; ; i++) {
      if(g == 0 && i >= s.size() && i >= b.s.size()) break;
      int x = g;
      if(i < s.size()) x += s[i];
      if(i < b.s.size()) x += b.s[i];
      c.s.push_back(x % BASE);
      g = x / BASE;
    }
    return c;
  }
};

ostream& operator << (ostream &out, const BigInteger& x) {
  out << x.s.back();
  for(int i = x.s.size()-2; i >= 0; i--) {
    char buf[20];
    sprintf(buf, "%08d", x.s[i]);
    for(int j = 0; j < strlen(buf); j++) out << buf[j];
  }
  return out;
}

istream& operator >> (istream &in, BigInteger& x) {
  string s;
  if(!(in >> s)) return in;
  x = s;
  return in;
}

#include<set>
#include<map>
set<BigInteger> s;
map<BigInteger, int> m;

int main() {
  BigInteger y;
  BigInteger x = y;
  BigInteger z = 123;

  BigInteger a, b;
  cin >> a >> b;
  cout << a + b << "\n";
  cout << BigInteger::BASE << "\n";
  return 0;
}

```

### 结构体（加减乘）

```cpp
# include <cstdio>
# include <iostream>
# include <cstring>
# include <algorithm>
# include <cmath>

using namespace std;

# define FOR(i, a, b) for(int i = a; i <= b; i++)
# define _FOR(i, a, b) for(int i = a; i >= b; i--)

struct BigInt
{
    static const int M = 1000;
    int num[M + 10], len;

    BigInt() { clean(); }	

	void clean(){
    	memset(num, 0, sizeof(num));
    	len = 1;
	}

    void read(){
    	char str[M + 10];
        scanf("%s", str);
        len = strlen(str);
        FOR(i, 1, len)
            num[i] = str[len - i] - '0';
    }

    void write(){
        _FOR(i, len, 1)
            printf("%d", num[i]);
        puts("");
    }
    
    void itoBig(int x){
    	clean();
    	while(x != 0){
    		num[len++] = x % 10;
    		x /= 10;
		}
		if(len != 1) len--;
	}

    bool operator < (const BigInt &cmp) const {
        if(len != cmp.len) return len < cmp.len;
        _FOR(i, len, 1)
            if(num[i] != cmp.num[i]) return num[i] < cmp.num[i];
        return false;
    }

    bool operator > (const BigInt &cmp) const { return cmp < *this; }
	bool operator <= (const BigInt &cmp) const { return !(cmp < *this); }
	bool operator != (const BigInt &cmp) const { return cmp < *this || *this < cmp; }
	bool operator == (const BigInt &cmp) const { return !(cmp < *this || *this < cmp); }

    BigInt operator + (const BigInt &A) const {
        BigInt S;
        S.len = max(len, A.len);
        FOR(i, 1, S.len){
            S.num[i] += num[i] + A.num[i];
            if(S.num[i] >= 10){
                S.num[i] -= 10;
                S.num[i + 1]++;
            }
        }
        while(S.num[S.len + 1]) S.len++;
        return S;
    }

    BigInt operator - (const BigInt &A) const {
        BigInt S;
        S.len = max(len, A.len);
        FOR(i, 1, S.len){
            S.num[i] += num[i] - A.num[i];
            if(S.num[i] < 0){
                S.num[i] += 10;
                S.num[i + 1]--;
            }
        }
        while(!S.num[S.len] && S.len > 1) S.len--;
        return S;
    }

    BigInt operator * (const BigInt &A) const {
        BigInt S;
        if((A.len == 1 && A.num[1] == 0) || (len == 1 && num[1] == 0)) return S;
        S.len = A.len + len - 1;
        FOR(i, 1, len)
            FOR(j, 1, A.len){
                S.num[i + j - 1] += num[i] * A.num[j];
                S.num[i + j] += S.num[i + j - 1] / 10;
                S.num[i + j - 1] %= 10;
            }
        while(S.num[S.len + 1]) S.len++;
        return S;
    }
};

int main()
{
	return 0;
}

```

### 函数解法（加、减、乘、比较）

```cpp
# include <cstdio>
# include <iostream>
# include <cstring>
# include <algorithm>
# include <cmath>

using namespace std;

# define FOR(i, a, b) for(int i = a; i <= b; i++)
# define _FOR(i, a, b) for(int i = a; i >= b; i--)

const int NR = 1000;

bool STRcompare(string str1, string str2){
	int len1 = str1.size(), len2 = str2.size();
    if(len1 != len2) return len1 < len2;
    FOR(i, 0, len1 - 1)
        if(str1[i] != str2[i]) return str1[i] < str2[i];
    return false;
}

string STRaddition(string str1, string str2){
    int sum[NR + 10], a[NR + 10], b[NR + 10];
    memset(sum, 0, sizeof(sum));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    int len1 = str1.size(), len2 = str2.size();
    FOR(i, 1, len1) a[i] = str1[len1 - i] - '0';
    FOR(i, 1, len2) b[i] = str2[len2 - i] - '0';
    int lenS = max(len1, len2);
    FOR(i, 1, lenS){
        sum[i] += a[i] + b[i];
    	if(sum[i] >= 10){
            sum[i] -= 10;
            sum[i + 1]++;
    	}
    }
    while(sum[lenS + 1]) lenS++;
    string ans;
    _FOR(i, lenS, 1) ans += sum[i] + '0';
    return ans;
}

string STRsubtraction(string str1, string str2){
    int sum[NR + 10], a[NR + 10], b[NR + 10];
    memset(sum, 0, sizeof(sum));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    if(STRcompare(str1, str2)) swap(str1, str2);
    int len1 = str1.size(), len2 = str2.size();
    FOR(i, 1, len1) a[i] = str1[len1 - i] - '0';
    FOR(i, 1, len2) b[i] = str2[len2 - i] - '0';
    int lenS = max(len1, len2);
    FOR(i, 1, lenS){
        sum[i] += a[i] - b[i];
    	if(sum[i] < 0){
            sum[i] += 10;
            sum[i + 1]--;
    	}
    }
    while(sum[lenS] == 0 && lenS > 1) lenS--;
    string ans;
    _FOR(i, lenS, 1) ans += sum[i] + '0';
    return ans;
}    

string STRmultiplication(string str1, string str2){
	if(str1.size() == 1 && str1[0] == '0') return str1;
	if(str2.size() == 1 && str2[0] == '0') return str2;
	int sum[NR + 10], a[NR + 10], b[NR + 10];
    memset(sum, 0, sizeof(sum));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    int len1 = str1.size(), len2 = str2.size();
    FOR(i, 1, len1) a[i] = str1[len1 - i] - '0';
    FOR(i, 1, len2) b[i] = str2[len2 - i] - '0';
    int lenS = len1 + len2 - 1;
    FOR(i, 1, len1)
        FOR(j, 1, len2){
            sum[i + j - 1] += a[i] * b[j];
            sum[i + j] += sum[i + j - 1] / 10;
        	sum[i + j - 1] %= 10;
        }
    while(sum[lenS + 1]) lenS++;
    string ans;
    _FOR(i, lenS, 1) ans += sum[i] + '0';
    return ans;
}

string char_to_string(char c){
	string str;
	str += c;
	return str;
}

string int_to_string(int x){
	int a[NR + 10], len = 0;
	while(x != 0){
		a[++len] = x % 10;
		x /= 10;
	}
	string str;
	_FOR(i, len, 1) str += a[i] + '0';
	if(len == 0) str = "0";
	return str;
}

int main()
{
	return 0;
}

```

### Java解法

```java
import java.util.Scanner;
 
public class BigInteger {
 
    public static void main(String[] args) {
        // TODO Auto-generated method stub
 
        Scanner cin = new Scanner(System.in);
        java.math.BigInteger a;
        java.math.BigInteger b;
        while(cin.hasNext()){
 
        a = cin.nextBigInteger();
 
        b = cin.nextBigInteger();
 
        System.out.println(a.add(b)); //大整数加法
 
        System.out.println(a.subtract(b)); //大整数减法
 
        System.out.println(a.multiply(b)); //大整数乘法
 
        System.out.println(a.divide(b)); //大整数除法(取整)
 
        System.out.println(a.remainder(b)); //大整数取模
 
        if( a.compareTo(b) == 0 ) System.out.println("a == b"); //大整数a==b
        else if( a.compareTo(b) > 0 ) System.out.println("a > b"); //大整数a>b
        else if( a.compareTo(b) < 0 ) System.out.println("a < b"); //大整数a<b
 
        System.out.println(a.abs()); //大整数a的绝对值
 
        int exponent=10;
 
        System.out.println(a.pow(exponent)); //大整数a的exponent次幂
 
        System.out.println(a.toString());
 
        //返回大整数p进制的字符串表示
        int p=8;
 
        System.out.println(a.toString(p));
        
    }
    }
}
```

### 完全解法

```cpp
/**********************************
高精度加，减，乘，除，取模，模板
**********************************/
#include <iostream>
#include <string>
using namespace std;
 
inline int compare(string str1, string str2)
{
      if(str1.size() > str2.size()) //长度长的整数大于长度小的整数
            return 1;
      else if(str1.size() < str2.size())
            return -1;
      else
            return str1.compare(str2); //若长度相等，从头到尾按位比较，compare函数：相等返回0，大于返回1，小于返回－1
}
//高精度加法
string ADD_INT(string str1, string str2)
{
      string MINUS_INT(string str1, string str2);
      int sign = 1; //sign 为符号位
      string str;
      if(str1[0] == '-') {
           if(str2[0] == '-') {
                 sign = -1;
                 str = ADD_INT(str1.erase(0, 1), str2.erase(0, 1));
           }else {
                 str = MINUS_INT(str2, str1.erase(0, 1));
           }
      }else {
           if(str2[0] == '-')
                 str = MINUS_INT(str1, str2.erase(0, 1));
           else {
                 //把两个整数对齐，短整数前面加0补齐
                 string::size_type l1, l2;
                 int i;
                 l1 = str1.size(); l2 = str2.size();
                 if(l1 < l2) {
                       for(i = 1; i <= l2 - l1; i++)
                       str1 = "0" + str1;
                 }else {
                       for(i = 1; i <= l1 - l2; i++)
                       str2 = "0" + str2;
                 }
                 int int1 = 0, int2 = 0; //int2 记录进位
                 for(i = str1.size() - 1; i >= 0; i--) {
                       int1 = (int(str1[i]) - 48 + int(str2[i]) - 48 + int2) % 10;  //48 为 '0' 的ASCII 码
                       int2 = (int(str1[i]) - 48 + int(str2[i]) - 48 +int2) / 10;
                       str = char(int1 + 48) + str;
                 }
                 if(int2 != 0) str = char(int2 + 48) + str;
          }
     }
     //运算后处理符号位
     if((sign == -1) && (str[0] != '0'))
          str = "-" + str;
     return str;
}
 
 
//高精度减法
string MINUS_INT(string str1, string str2)
{
     string MULTIPLY_INT(string str1, string str2);
     int sign = 1; //sign 为符号位
     string str;
     if(str2[0] == '-')
            str = ADD_INT(str1, str2.erase(0, 1));
     else {
            int res = compare(str1, str2);
            if(res == 0) return "0";
            if(res < 0) {
                  sign = -1;
                  string temp = str1;
                  str1 = str2;
                  str2 = temp;
            }
            string::size_type tempint;
            tempint = str1.size() - str2.size();
            for(int i = str2.size() - 1; i >= 0; i--) {
                 if(str1[i + tempint] < str2[i]) {
                       str1[i + tempint - 1] = char(int(str1[i + tempint - 1]) - 1);
                       str = char(str1[i + tempint] - str2[i] + 58) + str;
                 }
                 else
                       str = char(str1[i + tempint] - str2[i] + 48) + str;
            }
           for(int i = tempint - 1; i >= 0; i--)
                str = str1[i] + str;
     }
     //去除结果中多余的前导0
     str.erase(0, str.find_first_not_of('0'));
     if(str.empty()) str = "0";
     if((sign == -1) && (str[0] != '0'))
          str = "-" + str;
     return str;
}
 
//高精度乘法
string MULTIPLY_INT(string str1, string str2)
{
     int sign = 1; //sign 为符号位
     string str;
     if(str1[0] == '-') {
           sign *= -1;
           str1 = str1.erase(0, 1);
     }
     if(str2[0] == '-') {
           sign *= -1;
           str2 = str2.erase(0, 1);
     }
     int i, j;
     string::size_type l1, l2;
     l1 = str1.size(); l2 = str2.size();
     for(i = l2 - 1; i >= 0; i --) {  //实现手工乘法
           string tempstr;
           int int1 = 0, int2 = 0, int3 = int(str2[i]) - 48;
           if(int3 != 0) {
                  for(j = 1; j <= (int)(l2 - 1 - i); j++)
                        tempstr = "0" + tempstr;
                  for(j = l1 - 1; j >= 0; j--) {
                        int1 = (int3 * (int(str1[j]) - 48) + int2) % 10;
                        int2 = (int3 * (int(str1[j]) - 48) + int2) / 10;
                        tempstr = char(int1 + 48) + tempstr;
                  }
                  if(int2 != 0) tempstr = char(int2 + 48) + tempstr;
           }
           str = ADD_INT(str, tempstr);
     }
     //去除结果中的前导0
     str.erase(0, str.find_first_not_of('0'));
     if(str.empty()) str = "0";
     if((sign == -1) && (str[0] != '0'))
           str = "-" + str;
     return str;
}
//高精度除法
string DIVIDE_INT(string str1, string str2, int flag)
{
     //flag = 1时,返回商; flag = 0时,返回余数
     string quotient, residue; //定义商和余数
     int sign1 = 1, sign2 = 1;
     if(str2 == "0") {  //判断除数是否为0
           quotient = "ERROR!";
           residue = "ERROR!";
           if(flag == 1) return quotient;
           else return residue;
     }
     if(str1 == "0") { //判断被除数是否为0
           quotient = "0";
           residue = "0";
     }
     if(str1[0] == '-') {
           str1 = str1.erase(0, 1);
           sign1 *= -1;
           sign2 = -1;
     }
     if(str2[0] == '-') {
           str2 = str2.erase(0, 1);
           sign1 *= -1;
     }
     int res = compare(str1, str2);
     if(res < 0) {
           quotient = "0";
           residue = str1;
     }else if(res == 0) {
           quotient = "1";
           residue = "0";
     }else {
           string::size_type l1, l2;
           l1 = str1.size(); l2 = str2.size();
           string tempstr;
           tempstr.append(str1, 0, l2 - 1);
           //模拟手工除法
           for(int i = l2 - 1; i < l1; i++) {
                 tempstr = tempstr + str1[i];
                 for(char ch = '9'; ch >= '0'; ch --) { //试商
                       string str;
                       str = str + ch;
                       if(compare(MULTIPLY_INT(str2, str), tempstr) <= 0) {
                              quotient = quotient + ch;
                              tempstr = MINUS_INT(tempstr, MULTIPLY_INT(str2, str));
                              break;
                       }
                 }
           }
           residue = tempstr;
     }
     //去除结果中的前导0
     quotient.erase(0, quotient.find_first_not_of('0'));
     if(quotient.empty()) quotient = "0";
     if((sign1 == -1) && (quotient[0] != '0'))
     quotient = "-" + quotient;
     if((sign2 == -1) && (residue[0] != '0'))
     residue = "-" + residue;
     if(flag == 1) return quotient;
     else return residue;
}
 
//高精度除法,返回商
string DIV_INT(string str1, string str2)
{
      return DIVIDE_INT(str1, str2, 1);
}
//高精度除法,返回余数
string MOD_INT(string str1, string str2)
{
      return DIVIDE_INT(str1, str2, 0);
}
 
int main()
{
      char ch;
      string s1, s2, res;
      while(cin >> ch) {
    cin >> s1 >> s2;
           switch(ch) {
                case '+':  res = ADD_INT(s1, s2); break;   //高精度加法
                case '-':  res = MINUS_INT(s1, s2); break; //高精度减法
                case '*':  res = MULTIPLY_INT(s1, s2); break; //高精度乘法
                case '/':  res = DIV_INT(s1, s2); break; //高精度除法, 返回商
                case 'm':  res = MOD_INT(s1, s2); break; //高精度除法, 返回余数
                default :  break;
           }
           cout << res << endl;
      }
      return(0);
}
```

## 归并排序（求逆序对）

```cpp
int a[505000];
int aux[505000];
int n;
int ans;

void merge(int lo, int mid, int hi) {
    for (int k = lo; k <= hi; k++)
        aux[k] = a[k];

    int i = lo;
    int j = mid+1;

    for (int k = lo; k <= hi; k++) {
        if (i > mid) a[k] = aux[j++];
        else if (j > hi) a[k] = aux[i++];
        else if (aux[i] <= aux[j]) a[k] = aux[i++];
        else { a[k] = aux[j++]; ans += (mid - i +1); }
    }
}

void merge_sort(int lo, int hi) {
    if (lo >= hi)return;
    int mid = (lo + hi) / 2;
    merge_sort(lo, mid);
    merge_sort(mid + 1, hi);
    merge(lo, mid, hi);
}


signed main() {

#ifdef LOCAL
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%d", &a[i]);

    merge_sort(1, n);
    printf("%d\n", ans);

    return 0;
}
```

## 二分查找（库及模板）

```cpp
int BSearch(int R[],int low,int high,int val)
{
    while(low<=high)
    {
        int mid=(low+high)/2;
        if(R[mid]==val) return mid;
        else if(R[mid]>val)
            high=mid-1;
        else low=mid+1;
    }
    return -1;
}
```

lower_bound( )和upper_bound( )都是利用二分查找的方法在一个排好序的数组中进行查找的。

在从小到大的排序数组中，

lower_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于或等于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

upper_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个大于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

在从大到小的排序数组中，重载lower_bound()和upper_bound()

lower_bound( begin,end,num,greater<type>() ):从数组的begin位置到end-1位置二分查找第一个小于或等于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

upper_bound( begin,end,num,greater<type>() ):从数组的begin位置到end-1位置二分查找第一个小于num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

## 字符串hash

### 算法

```typedef unsigned long long ull;```

```char s[10010];```



```ull base = 131;```

```int prime = 233317;```

```ull mod = 212370440130137957ll;```



```ans = (ans * base + (ull) s[i]) % mod + prime;```



### 模板题

洛谷P3370

```cpp
#include<iostream>
#include<cstring>
#include<algorithm>
#include<cstdio>

using namespace std;
typedef unsigned long long ull;
ull base = 131;
ull a[10010];
char s[10010];
int n, ans = 1;
int prime = 233317;
ull mod = 212370440130137957ll;

ull hashe(char s[]) {
    int len = strlen(s);
    ull ans = 0;
    for (int i = 0; i < len; i++)
        ans = (ans * base + (ull) s[i]) % mod + prime;
    return ans;
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        scanf("%s", s);
        a[i] = hashe(s);
    }
    sorsasdt(a + 1, a + n + 1);
    for (int i = 1; i < n; i++) {
        if (a[i] != a[i + 1])
            ans++;
    }
    printf("%d", ans);
} 
```