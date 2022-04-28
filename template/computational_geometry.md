# 关于

# 前置知识

尽量减少使用除法、三角函数、反三角函数，以增加精度减少时间。

$pi = acos(-1);$

余弦定理 $c^2 = a^2 + b^2 - 2abcos\theta$

正弦定理 $\frac{a}{sin\alpha} = \frac{b}{sin\beta} = \frac{c}{sin\theta}$

# 浮点数比较
```cpp
const double eps = 1e-8;

int sign(double x) { //符号函数
    if(fabs(x) < eps) return 0;
    if(x < 0) return -1;
    return 1;
}

int cmp(double x, double y) { // 比较函数
    if(fabs(x - y) < eps) return 0;
    if(x < y) return -1;
    return 1;
}
```

# 向量
```cpp
struct Point {
    double x, y;
};
using Vector = Point;
Vector operator + (Vector A, Vector B) { return Vector{A.x+B.x, A.y+B.y};}
Vector operator - (Point A, Point B) { return Vector{A.x-B.x, A.y-B.y};}
Vector operator * (Vector A, double p) { return Vector{A.x*p, A.y*p};}
Vector operator / (Vector A, double p) { return Vector{A.x/p, A.y/p};}

// 点积
double dot(Vector a, Vector b) {
    return a.x * b.x + a.y * b.y;
}

// 叉积
double cross(Vector a, Vector b) {
    return a.x * b.y - b.x * a.y;
}

// 模长
double get_length(Vector a) {
    return sqrt(dot(a, a));
}

// 向量夹角（余弦定理）
double get_angle(Vector a, Vector b) {
    return acos(dot(a, b) / get_length(a) / get_length(b));
}

// 两个向量平行四边形面积
double area(Point a, Point b, Point c) {
    return cross(b - a, c - a);
}

// 逆时针旋转
Vector rotate(Vector a, double angle) {
    return Vector{a.x * cos(angle) - a.y * sin(angle), a.y * cos(angle) + a.x * sin(angle)};
}
```

# 极角排序
根据叉乘排序
```cpp
const double eps = 1e-8;
int sign(double x) {
    if(fabs(x) < eps) return 0;
    return x > 0 ? 1 : -1;
}

struct Point {
    double x, y;
    int quad() const { return sign(y) == 1 || (sign(y) == 0 && sign(x) <= 0);}
};
using Vector = Point;
 
double cross(Vector a, Vector b) {
    return a.x * b.y - a.y * b.x;
}
 
bool operator < (Vector a, Vector b) {
    if(a.quad() != b.quad()) return a.quad() < b.quad();
    return sign(cross(a, b)) > 0;
}
```

根据atan2排序
```cpp
const double eps = 1e-16;
int cmp(double x, double y) {
    if(fabs(x - y) < eps) return 0;
    return x < y ? -1 : 1;
}

struct Point {
    double x, y;
};
using Vector = Point;

bool operator < (Vector a, Vector b) {
    return cmp(atan2(a.y, a.x), atan2(b.y, b.x)) < 0;
}
```

# 点与线

1. 一般式 $ax + by + c = 0$
2. 点向式 $p_0 + t\vec{v}$
3. 斜截式 $y = kx + b$


```cpp
// 判断点在直线上（叉乘为0）
bool on_line (Point p, Point a, Point b) {
    Vector v1 = p - a, v2 = b - a;
    return sign(cross(v1, v2)) == 0;
}

// 两直线交点
// cross(v, w) == 0则两直线平行或者重合
Point line_intersection(Point p, Vector v, Point q, Vector w) {
    Vector u = p - q;
    double t = cross(w, u) / cross(v, w);
    return p + v * t;
}

// 点到直线的距离
double distance_to_line(Point p, Point a, Point b) {
    Vector v1 = b - a, v2 = p - a;
    return fabs(cross(v1, v2) / get_length(v1));
}

// 点到线段的距离
double distance_to_segment(Point p, Point a, Point b) {
    if (a == b) return get_length(p - a);
    Vector v1 = b - a, v2 = p - a, v3 = p - b;
    if (sign(dot(v1, v2)) < 0) return get_length(v2);
    if (sign(dot(v1, v3)) > 0) return get_length(v3);
    return distance_to_line(p, a, b);
}

// 点在直线上的投影
double get_line_projection(Point p, Point a, Point b) {
    Vector v = b - a;
    return a + v * (dot(v, p - a) / dot(v, v));
}

// 点是否在线段上
bool on_segment(Point p, Point a, Point b) {
    return sign(cross(p - a, p - b)) == 0 && sign(dot(p - a, p - b)) <= 0;
}

// 判断两线段是否相交
bool segment_intersection(Point a1, Point a2, Point b1, Point b2) {
    double c1 = cross(a2 - a1, b1 - a1), c2 = cross(a2 - a1, b2 - a1);
    double c3 = cross(b2 - b1, a2 - b1), c4 = cross(b2 - b1, a1 - b1);
    return sign(c1) * sign(c2) <= 0 && sign(c3) * sign(c4) <= 0;
}
```


# 三角形

## 面积

叉积

海伦公式

$p = (a + b + c) / 2$

$S = sqrt(p \cdot (p - a) \cdot (p - b) \cdot (p - c))$

## 三角形四心

1. 外心，外接圆圆心：三边中垂线交点。到三角形三个顶点的距离相等
2. 内心，内切圆圆心：角平分线交点，到三边距离相等      
3. 垂心：三条垂线交点
4. 重心：三条中线交点（到三角形三顶点距离的平方和最小的点，三角形内到三边距离之积最大的点）


# 多边形

```cpp
// 多边形面积
double polygon_area(Point p[], int n) {
    double s = 0;
    for (int i = 1; i + 1 < n; i ++ )
        s += cross(p[i] - p[0], p[i + 1] - p[i]);
    return s / 2;
}

// 多边形点包含 2-点在线上 1-点在多边形内 0-点在多边形外
int contain(vector<Point>& polygon, Point p) {
    int ret = 0, n = polygon.size();
    for(int i = 0; i < n; ++ i) {
        Point u = polygon[i], v = polygon[(i + 1) % n];
        if (on_segment(p, u, v)) return 2;
        if (sign(u.y - v.y) <= 0) swap(u, v);
        if (sign(p.y - u.y) > 0 || sign(p.y - v.y) <= 0) continue;
        ret += sign(cross(p - u, v - u)) > 0;
    }
    return ret & 1;
}
```

## 皮克定理
$S = a + b/2 - 1$

皮克定理是指一个计算点阵中顶点在格点上的多边形面积公式.

其中a表示多边形内部的点数，b表示多边形边界上的点数，S表示多边形的面积。


# 圆
```cpp
struct Circle {
	Point o; double r;
};

// 求圆与直线的交点
bool circle_line_intersection(Circle a, Line l, Point &p1, Point &p2) { 
	double x = dot(l.a - a.o, l.b - l.a),
		y = (l.b - l.a).len2(),
		d = x * x - y * ((l.a - a.o).len2() - a.r * a.r);
	if (sign(d) < 0) return false;
	d = max(d, 0.0);
	Point p = l.a - ((l.b - l.a) * (x / y)), delta = (l.b - l.a) * (sqrt(d) / y);
	p1 = p + delta, p2 = p - delta;
	return true;
}

// 求圆与圆的交面积
double circle_circle_intersection_area(const Circle &c1, const Circle &c2) {
	double d = get_length((c1.o - c2.o));
	if (sign(d - (c1.r + c2.r)) >= 0) return 0;
	if (sign(d - abs(c1.r - c2.r)) <= 0) {
		double r = min(c1.r, c2.r);
		return r * r * pi;
	}
	double x = (d * d + c1.r * c1.r - c2.r * c2.r) / (2.0 * d),
		   t1 = acos(x / c1.r), t2 = acos((d - x) / c2.r);
    return c1.r * c1.r * t1 + c2.r * c2.r * t2 - (c1.r * c1.r * sinl(t1 * 2) + c2.r * c2.r * sinl(t2 * 2)) / 2.0;
	// return c1.r * c1.r * t1 + c2.r * c2.r * t2 - d * c1.r * sin(t1); // d和c1.r过大而t1过小会被卡
}

// 求圆与圆的交点，注意调用前要先判定重圆
bool circle_circle_intersection(Circle a, Circle b, Point &p1, Point &p2) { 
	double s1 = (a.o - b.o).len();
	if (sign(s1 - a.r - b.r) > 0 || sign(s1 - abs(a.r - b.r)) < 0) return false;
	double s2 = (a.r * a.r - b.r * b.r) / s1;
	double aa = (s1 + s2) * 0.5, bb = (s1 - s2) * 0.5;
	Point o = (b.o - a.o) * (aa / (aa + bb)) + a.o;
	Point delta = (b.o - a.o).unit().turn90() * newSqrt(a.r * a.r - aa * aa);
	p1 = o + delta, p2 = o - delta;
	return true;
}

// 求点到圆的切点，按关于点的顺时针方向返回两个点
bool tan_circle_point(const Circle &c, const Point &p0, Point &p1, Point &p2) {
	double x = (p0 - c.o).len2(), d = x - c.r * c.r;
	if (d < EPS) return false; // 点在圆上认为没有切点
	Point p = (p0 - c.o) * (c.r * c.r / x);
	Point delta = ((p0 - c.o) * (-c.r * sqrt(d) / x)).turn90();
	p1 = c.o + p + delta;
	p2 = c.o + p - delta;
	return true;
}

// 求圆到圆的外共切线，按关于 c1.o 的顺时针方向返回两条线
vector<Line> ciecle_circle_extan(const Circle &c1, const Circle &c2) {
	vector<Line> ret;
	if (sign(c1.r - c2.r) == 0) {
		Point dir = c2.o - c1.o;
		dir = (dir * (c1.r / dir.len())).turn90();
		ret.push_back(Line(c1.o + dir, c2.o + dir));
		ret.push_back(Line(c1.o - dir, c2.o - dir));
	} else {
		Point p = (c1.o * -c2.r + c2.o * c1.r) / (c1.r - c2.r);
		Point p1, p2, q1, q2;
		if (tanCP(c1, p, p1, p2) && tanCP(c2, p, q1, q2)) {
			if (c1.r < c2.r) swap(p1, p2), swap(q1, q2);
			ret.push_back(Line(p1, q1));
			ret.push_back(Line(p2, q2));
		}
	}
	return ret;
}

// 求圆到圆的内共切线，按关于 c1.o 的顺时针方向返回两条线
vector<Line> ciecle_circke_intan(const Circle &c1, const Circle &c2) {
	vector<Line> ret;
	Point p = (c1.o * c2.r + c2.o * c1.r) / (c1.r + c2.r);
	Point p1, p2, q1, q2;
	if (tanCP(c1, p, p1, p2) && tanCP(c2, p, q1, q2)) { // 两圆相切认为没有切线
		ret.push_back(Line(p1, q1));
		ret.push_back(Line(p2, q2));
	}
	return ret;
}

```

# 凸包

```cpp
// 以横坐标为第一关键字，纵坐标为第二关键字排序
// 会改变输入点的顺序
// < 0 边上可以有点, <= 0 则不能

// double
const double eps = 1e-6;
int sign(double x) {
    if(fabs(x) < eps) return 0;
    return x < 0 ? -1 : 1;
}

// point
struct Point {
    double x, y;
};
using Vector = Point;
Vector operator - (Vector a, Vector b) {return Vector{a.x - b.x, a.y - b.y}; }
bool operator < (Vector a, Vector b) {return a.x < b.x || a.x == b.x && a.y < b.y; }
double cross(Vector a, Vector b) { return a.x * b.y - a.y * b.x; }

// 点数大于等于2且全部重合时，返回两个相同的点
// convex hull
vector<Point> andrew(vector<Point> &s) {
    sort(s.begin(), s.end());
    vector<Point> ret(s.size() * 2);
    int sz = 0;
    for(int i = 0; i < s.size(); i++) {
        while(sz > 1 && sign(cross(ret[sz - 1] - ret[sz - 2], s[i] - ret[sz - 2])) <= 0) sz--;
        ret[sz++] = s[i];
    }
    int k = sz;
    for(int i = s.size() - 2; i >= 0; i--) {
        while(sz > k && sign(cross(ret[sz - 1] - ret[sz - 2], s[i] - ret[sz - 2])) <= 0) sz--;
        ret[sz++] = s[i];
    }
    ret.resize(sz - (s.size() > 1));
    return ret;
}
```

# 半平面交
```cpp
// double
const double eps = 1e-8;
int sign(double x) {
    if(fabs(x) < eps) return 0;
    return x < 0 ? -1 : 1;
}
int cmp(double x, double y) {
    return sign(x - y);
}

// point
struct Point {
    double x, y;
    int quad() const {return sign(y) == 1 || (sign(y) == 0 && sign(x) <= 0);}
};
using Vector = Point;
Vector operator + (Vector a, Vector b) { return Vector{a.x + b.x, a.y + b.y}; }
Vector operator - (Vector a, Vector b) { return Vector{a.x - b.x, a.y - b.y}; }
Vector operator * (Vector a, double p) { return Vector{a.x * p, a.y * p}; }
Vector operator / (Vector a, double p) { return Vector{a.x / p, a.y / p}; }
double dot(Vector a, Vector b) { return a.x * b.x + a.y * b.y; }
double cross(Vector a, Vector b) { return a.x * b.y - a.y * b.x; }
bool operator < (Vector a, Vector b) { if(a.quad() != b.quad()) return a.quad() < b.quad(); return sign(cross(a, b)) > 0; }

// line
struct Line {
    Point s, t;
    bool include(const Point &p) const {return sign(cross(t - s, p - s)) > 0;}
};
Point line_intersection(const Line& a, const Line& b) {
    double s1 = cross(a.t - a.s, b.s - a.s), s2 = cross(a.t - a.s, b.t - a.s);
    return (b.s * s2 - b.t * s1) / (s2 - s1);
}
bool parallel(Line a, Line b) { return !sign(cross(a.t - a.s, b.t - b.s)); }
bool same_dir(Line a, Line b) { return parallel(a, b) && sign(dot(a.t - a.s, b.t - b.s)) == 1; }
bool operator < (Line a, Line b) { if(same_dir(a, b)) return b.include(a.s); else return (a.t - a.s) < (b.t - b.s); }
bool check(const Line u, Line v, Line w) {return w.include(line_intersection(u, v)); }


// 会更改l的顺序
// 两条直线重合自动忽略
// q 保存半平面交上的直线（多线交于一点时，这些直线都被保留）
// ret 保存半平面交直线的的交点（多线交于一点时，会产生重复点）
// half plane intersection
vector<Point> half_plane_intersection(vector<Line> &l) {
    sort(l.begin(), l.end());
    deque<Line> q;
    for(int i = 0; i<l.size(); i++) {
        if(i && same_dir(l[i], l[i-1])) continue;
        while(q.size() > 1 && !check(q[q.size()-2], q[q.size()-1], l[i])) q.pop_back();
        while(q.size() > 1 && !check(q[1], q[0], l[i])) q.pop_front();
        q.push_back(l[i]);
    }
    while(q.size() > 2 && !check(q[q.size()-2], q[q.size()-1], q[0])) q.pop_back();
    while(q.size() > 2 && !check(q[1], q[0], q[q.size()-1])) q.pop_front();
    vector<Point> ret;
    for(int i = 0; i<q.size(); i++) ret.push_back(line_intersection(q[i], q[(i+1)%q.size()]));
    return ret;
}
```