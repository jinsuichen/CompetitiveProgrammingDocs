# 关于

包含快速排序、归并排序、二分、三分、模拟退火、高精度、前缀和与差分、位运算、区间合并。

待完善的算法有双指针、离散化。

# 快速排序

对于长度为n的数组，```l```为 0，```r```为 n-1

```cpp
void quick(int l, int r){

    if(l >= r) return;

    int x = a[(l+r)>>1], i = l-1, j = r+1;
    while(i < j){

        while(a[++i]<x);
        while(a[--j] > x);
        if( i < j) swap(a[i], a[j]);

    }

    quick(l, j);
    quick(j+1, r);

}
```

# 归并排序

对于长度为n的数组，```l```为 0，```r```为 n-1

```cpp
void merge_sort(int l, int r){

    if(l >= r) return;

    int mid = l + r >> 1;

    merge_sort(l, mid);
    merge_sort(mid+1, r);

    int i = l, j = mid+1; int p = l;
    while(i<=mid && j <= r){
        if(a[i] <= a[j]) tmp[p++] = a[i++];
        else tmp[p++] = a[j++];
    }

    while(i <= mid) tmp[p++] = a[i++];
    while(j <= r) tmp[p++] = a[j++];

    for(int i = l; i<=r; i++ ){
        a[i] = tmp[i];
    }
}
```

求逆序对。对于长度为n的数组，```l```为 0，```r```为 n-1

```cpp
long long cnt = 0;

void merge_sort(int l, int r){

    if( l >=r) return;

    int mid = l + r >> 1;

    merge_sort(l, mid);
    merge_sort(mid+1, r);

    int i = l, j = mid+1; int p = l;
    while(i<=mid && j <= r){
        if(a[i] <= a[j]) tmp[p++] = a[i++];
        else tmp[p++] = a[j++], cnt+=mid-i+1;
    }

    while(i <= mid) tmp[p++] = a[i++];
    while(j <= r) tmp[p++] = a[j++];

    for(int i = l; i<=r; i++ ){
        a[i] = tmp[i];
    }
}
```

# 二分

红色性质---绿色性质

区间[l, r]被划分成[l, mid]和[mid+1, r]时使用。满足绿色性质的最左点。

可以实现lower_bound和upper_bound

```cpp
int bsearch_1(int l, int r){
    while(l < r){
        int mid = l + r >> 1;
        if(check(mid)) r = mid;
        else l = mid+1;
    }
    return l;
}
```

区间[l, r]被划分成[l, mid-1]和[mid, r]时使用。满足红色性质的最右点。

```cpp
int bsearch_2(int l, int r){
    while(l < r){
        int mid = l + r + 1 >> 1;
        if(check(mid)) l = mid;
        else r = mid-1;
    }
    return l;
}
```

# 三分

答案是[l, r]之间的值，需要再将[l, r]check一遍。

1. \>  极小值点，区间左部分
2. \>= 极小值点，区间右部分
3. <  极大值点，区间左部分
3. <= 极大值，区间右部分

```cpp
while(r - l > 3) {
    int len = (r - l + 1) / 3 ;
    int lmid = l + len , rmid = r - len ;
    if(func(lmid) > func(rmid)) l = lmid;
    else  r = rmid;
}
```

# 模拟退火

```cpp
mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
double rd(double l, double r) {
    uniform_real_distribution<double> u(l, r);
    return u(rng);
}

double ans = 1e300;
double calc(double x) {
    double ret = 0;
    // calculate...
    ans = min(ans, ret);
    return ret;
}

void sumulate_anneal() {
    double cur = rd(0, 10000); // 在可行域中随机选一个值
    // t初始化为可跳转范围，要覆盖可行域
    for(double t = 1e4; t > 1e-4; t*=0.99) { 
        double nxt = rd(cur - t, cur + t);
        double dt = calc(nxt) - calc(cur);
        if(exp(-dt / t) > rd(0, 1)) cur = nxt; // 求最大值将dt的负号去掉
    }
}
```

# 高精度

支持加法和乘法

```cpp
#include <iostream>
#include <cstring>

using namespace std;

struct BigInt {
    const static int mod = 10000;
    const static int DLEN = 4;
    int a[600], len;

    BigInt() {
        memset(a, 0, sizeof(a));
        len = 1;
    }

    BigInt(int v) {
        memset(a, 0, sizeof(a));
        len = 0;
        do {
            a[len++] = v % mod;
            v /= mod;
        } while (v);
    }

    BigInt(const char s[]) {
        memset(a, 0, sizeof(a));
        int L = strlen(s);
        len = L / DLEN;
        if (L % DLEN)len++;
        int index = 0;
        for (int i = L-1;
             i >= 0;
             i -= DLEN){
            int t = 0;
            int k = i - DLEN + 1;
            if (k < 0)k = 0;
            for (int j = k; j <= i; j++)
                t = t * 10 + s[j] - '0';
            a[index++] = t;
        }
    }

    BigInt operator+(const BigInt &b) const {
        BigInt res;
        res.len = max(len, b.len);
        for (int i = 0; i <= res.len; i++)
            res.a[i] = 0;
        for (int i = 0; i < res.len; i++) {
            res.a[i] += ((i < len) ? a[i] : 0) + ((i < b.len) ? b.a[i] : 0);
            res.a[i + 1] += res.a[i] / mod;
            res.a[i] %= mod;
        }
        if (res.a[res.len] > 0)res.len++;
        return res;


    }

    BigInt operator*(const BigInt &b) const {
        BigInt res;
        for (int i = 0; i < len; i++) {
            int up = 0;
            for (int j = 0; j < b.len; j++) {
                int temp = a[i] * b.a[j] + res.a[i + j] + up;
                res.a[i + j] = temp % mod;
                up = temp / mod;
            }
            if (up != 0)
                res.a[i + b.len] = up;
        }
        res.len = len + b.len;
        while (res.a[res.len - 1] == 0 && res.len > 1)res.len--;
        return res;
    }

    void output() {
        printf("%d", a[len-1]);
        for (int i = len-2;
             i >= 0;
             i--)
            printf("%04d", a[i]);
        printf("\n");
    }
};
```

完全大数模板

输入 ```cin>>a```

输出 ```a.print();```

注意这个输入不能自动去掉前导 0 的，可以先读入到 char 数组，去掉前导 0，再用构造函数。

```cpp
#include <iostream>
#include <cstring>

using namespace std;
/*
* 完全大数模板
* 输入 cin>>a
* 输出 a.print();
* 注意这个输入不能自动去掉前导 0 的，可以先读入到 char 数组，去掉前导 0，再用构造函数。
*/
#define MAXN 9999
#define MAXSIZE 1010
#define DLEN 4

class BigNum {
private:
    int a[500]; //可以控制大数的位数
    int len;
public:
    BigNum() {
        len = 1;
        memset(a, 0, sizeof(a));
    } //构造函数
    BigNum(const int); //将一个 int 类型的变量转化成大数
    BigNum(const char *); //将一个字符串类型的变量转化为大数
    BigNum(const BigNum &); //拷贝构造函数
    BigNum &operator=(const BigNum &); //重载赋值运算符，大数之间进行赋值运算
    friend istream &operator>>(istream &, BigNum &); //重载输入运算符
    friend ostream &operator<<(ostream &, BigNum &); //重载输出运算符



    BigNum operator+(const BigNum &) const; //重载加法运算符，两个大数之间的相加运算
    BigNum operator-(const BigNum &) const; //重载减法运算符，两个大数之间的相减运算
    BigNum operator*(const BigNum &) const; //重载乘法运算符，两个大数之间的相乘运算
    BigNum operator/(const int &) const; //重载除法运算符，大数对一个整数进行相除运算

    BigNum operator^(const int &) const; //大数的 n 次方运算
    int operator%(const int &) const; //大数对一个类型的变量进行取模运算int
    bool operator>(const BigNum &T) const; //大数和另一个大数的大小比较
    bool operator>(const int &t) const; //大数和一个 int 类型的变量的大小比较

    void print(); //输出大数
};

//将一个 int 类型的变量转化为大数
BigNum::BigNum(const int b) {
    int c, d = b;
    len = 0;
    memset(a, 0, sizeof(a));
    while (d > MAXN) {
        c = d - (d / (MAXN + 1)) * (MAXN + 1);
        d = d / (MAXN + 1);
        a[len++] = c;
    }
    a[len++] = d;
}

//将一个字符串类型的变量转化为大数
BigNum::BigNum(const char *s) {
    int t, k, index, L, i;
    memset(a, 0, sizeof(a));
    L = strlen(s);
    len = L / DLEN;
    if (L % DLEN)len++;
    index = 0;
    for (i = L - 1; i >= 0; i -= DLEN) {
        t = 0;
        k = i - DLEN + 1;
        if (k < 0)k = 0;
        for (int j = k; j <= i; j++)
            t = t * 10 + s[j] - '0';
        a[index++] = t;
    }
}

//拷贝构造函数
BigNum::BigNum(const BigNum &T) : len(T.len) {


    int i;
    memset(a, 0, sizeof(a));
    for (i = 0; i < len; i++)
        a[i] = T.a[i];
}

//重载赋值运算符，大数之间赋值运算
BigNum &BigNum::operator=(const BigNum &n) {
    int i;
    len = n.len;
    memset(a, 0, sizeof(a));
    for (i = 0; i < len; i++)
        a[i] = n.a[i];
    return *this;
}

istream &operator>>(istream &in, BigNum &b) {
    char ch[MAXSIZE * 4];
    int i = -1;
    in >> ch;
    int L = strlen(ch);
    int count = 0, sum = 0;
    for (i = L - 1; i >= 0;) {
        sum = 0;
        int t = 1;
        for (int j = 0; j < 4 && i >= 0; j++, i--, t *= 10) {
            sum += (ch[i] - '0') * t;
        }
        b.a[count] = sum;
        count++;
    }
    b.len = count++;
    return in;
}

//重载输出运算符
ostream &operator<<(ostream &out, BigNum &b) {
    int i;
    cout << b.a[b.len - 1];
    for (i = b.len - 2; i >= 0; i--) {
        printf("%04d", b.a[i]);
    }
    return out;
}

//两个大数之间的相加运算
BigNum BigNum::operator+(const BigNum &T) const {
    BigNum t(*this);
    int i, big;
    big = T.len > len ? T.len : len;
    for (i = 0; i < big; i++) {
        t.a[i] += T.a[i];
        if (t.a[i] > MAXN) {
            t.a[i + 1]++;
            t.a[i] -= MAXN + 1;
        }
    }
    if (t.a[big] != 0)
        t.len = big + 1;
    else t.len = big;
    return t;
}

//两个大数之间的相减运算
BigNum BigNum::operator-(const BigNum &T) const {
    int i, j, big;
    bool flag;
    BigNum t1, t2;
    if (*this > T) {
        t1 = *this;
        t2 = T;
        flag = 0;
    } else {
        t1 = T;
        t2 = *this;
        flag = 1;
    }
    big = t1.len;
    for (i = 0; i < big; i++) {
        if (t1.a[i] < t2.a[i]) {
            j = i + 1;
            while (t1.a[j] == 0)
                j++;
            t1.a[j--]--;
            while (j > i)
                t1.a[j--] += MAXN;
            t1.a[i] += MAXN + 1 - t2.a[i];
        } else t1.a[i] -= t2.a[i];
    }
    t1.len = big;
    while (t1.a[t1.len - 1] == 0 && t1.len > 1) {
        t1.len--;
        big--;
    }
    if (flag)
        t1.a[big - 1] = 0 - t1.a[big - 1];
    return t1;
}

//两个大数之间的相乘
BigNum BigNum::operator*(const BigNum &T) const {
    BigNum ret;
    int i, j, up;
    int temp, temp1;
    for (i = 0; i < len; i++) {
        up = 0;
        for (j = 0; j < T.len; j++) {
            temp = a[i] * T.a[j] + ret.a[i + j] + up;
            if (temp > MAXN) {
                temp1 = temp - temp / (MAXN + 1) * (MAXN + 1);
                up = temp / (MAXN + 1);
                ret.a[i + j] = temp1;
            } else {
                up = 0;
                ret.a[i + j] = temp;
            }
        }
        if (up != 0)
            ret.a[i + j] = up;
    }
    ret.len = i + j;
    while (ret.a[ret.len - 1] == 0 && ret.len > 1)ret.len--;
    return ret;
}

//大数对一个整数进行相除运算
BigNum BigNum::operator/(const int &b) const {
    BigNum ret;
    int i, down = 0;
    for (i = len - 1; i >= 0; i--) {
        ret.a[i] = (a[i] + down * (MAXN + 1)) / b;
        down = a[i] + down * (MAXN + 1) - ret.a[i] * b;
    }
    ret.len = len;
    while (ret.a[ret.len - 1] == 0 && ret.len > 1)
        ret.len--;
    return ret;
}

//大数对一个 int 类型的变量进行取模
int BigNum::operator%(const int &b) const {
    int i, d = 0;
    for (i = len - 1; i >= 0; i--)
        d = ((d * (MAXN + 1)) % b + a[i]) % b;
    return d;
}

//大数的 n 次方运算
BigNum BigNum::operator^(const int &n) const {
    BigNum t, ret(1);
    int i;
    if (n < 0)exit(-1);
    if (n == 0)return 1;
    if (n == 1)return *this;
    int m = n;
    while (m > 1) {
        t = *this;
        for (i = 1; (i << 1) <= m; i <<= 1)
            t = t * t;
        m -= i;
        ret = ret * t;
        if (m == 1)ret = ret * (*this);
    }
    return ret;
}

//大数和另一个大数的大小比较
bool BigNum::operator>(const BigNum &T) const {
    int ln;
    if (len > T.len)return true;
    else if (len == T.len) {
        ln = len - 1;
        while (a[ln] == T.a[ln] && ln >= 0)
            ln--;
        if (ln >= 0 && a[ln] > T.a[ln])
            return true;
        else
            return false;
    } else
        return false;
}

//大数和一个 int 类型的变量的大小比较
bool BigNum::operator>(const int &t) const {
    BigNum b(t);
    return *this > b;
}

//输出大数
void BigNum::print() {
    int i;
    printf("%d", a[len - 1]);
    for (i = len - 2; i >= 0; i--)
        printf("%04d", a[i]);
    printf("\n");
}
```

# 前缀和与差分

二维差分

```cpp
int a[1005][1005];
int d[1005][1005];

void insert(int x1, int y1, int x2, int y2, int c){
    d[x1][y1] += c;
    d[x2+1][y1] -= c;
    d[x1][y2+1] -= c;
    d[x2+1][y2+1] += c;
}
```

# 位运算

二进制中1的个数

```cpp
int lowbit(int x){
    return x & -x;
}
int solve(int x){
    int cnt = 0;
        while(x){
            cnt++;
            x -= lowbit(x);
        }
    return cnt;
}
```

# 区间合并

```cpp
int n;
vector<pair<int, int> > v;

void read(){
    cin >> n;
    for(int i = 0; i<n; i++){
        int a, b; scanf("%d%d", &a, &b);
        v.push_back({a, b});
    }
}

vector<pair<int, int>> solve(){
    sort(v.begin(), v.end());
    vector<pair<int, int> > ans;
    for(auto seg : v){
        if(ans.empty() || ans.back().second  < seg.first) ans.push_back(seg);
        else ans.back().second = max(seg.second, ans.back().second);
    }
    return ans;
}
```

