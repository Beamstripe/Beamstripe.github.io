---
title: '2022 Summer ACM training-HDU Vol.8'
date: 2022-08-13
permalink: /posts/2022/08/2022SummerTraining15-HDU8/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

![image-20220811200730831](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220811200730831.png)

# 1001 [Theramore](http://acm.hdu.edu.cn/showproblem.php?pid=7220)

签到题：处理字符串得到最小字典序串

```c++
#include <iostream>
#include <algorithm>
#include<cstdio>
using namespace std;
#define ll long long 
int main(){
    int t;cin>>t;
	while(t--){
		string s;cin>>s;
		int a[100005],b[100005];
		int k=0,ks=0;
		for(int i=0;i<s.length();i++){
			if(i%2==0) a[k++]=s[i]-'0';
			else b[ks++]=s[i]-'0';
		}
		
		sort(a,a+k);
		sort(b,b+ks);
		for(int i=0;i<s.length();i++){
			if(i%2==0) cout<<a[i/2];
			else cout<<b[i/2];
		}
		cout<<endl;
	} 
}

```

# 1004 [Quel'Thalas](http://acm.hdu.edu.cn/showproblem.php?pid=7223)

签到题：答案为 $2n$

# 1005 [Ironforge](http://acm.hdu.edu.cn/showproblem.php?pid=7224)

一条链，每个点上有一个数 $a_i$ ，每条边上有一个质数 $b_i$ 。一开始在某个点上，有一个空背包，走到一个点上可以把它的质因子放进背包，一条边如果背包里有那个质数就可以走。多组询问求从 $x$ 出发能否走到 $y$（即求每个点能走到的最大范围）。

# 1007 [Darnassus](http://acm.hdu.edu.cn/showproblem.php?pid=7226)

给出一个排列 $p$，把每个位置视为点，建一个无向图，$i,j$ 之间的边权为 $|i-j|*|p_i-p_j|$。求这个图的最小生成树。

由于朴素建稠密图的方法将有 $\frac{n(n-1)}2$ 条边，考虑排除掉一些不满足条件的边。

考虑将相邻顶点建边，即连一条从 $1$ 到 $n$ 的链，得到边权为 $|p_i-p_{i-1}|\le n-1$ ，推知对最小生成树中也只有边权 $|i-j|\times |p_i-p_j|\leq n-1$ 的边（Kruskal 算法的原理），意味着 $|i-j|$ 和 $|p_i-p_j|$ 必有至少一个 $\leq \sqrt{n-1}$

因此可以在 $[i,i+\sqrt n]$ 的区间内搜索所有边(在下标与排列 $p$ 中)，再利用Kruskal求出最小生成树即可

```c++
#include<bits/stdc++.h>
#define mset(a, b) memset(a, b, sizeof(a))
#define mcpy(a, b) memcpy(a, b, sizeof(a))
using namespace std;
typedef long long LL;
const int MAXN = 50005;

template <typename T> inline void read(T &WOW) {
    T x = 0, flag = 1;
    char ch = getchar();
    while (!isdigit(ch)) {
        if (ch == '-') flag = -1;
        ch = getchar();
    }
    while (isdigit(ch)) {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    WOW = flag * x;
}

int n, p[MAXN], pos[MAXN], ufs[MAXN];

int getf(int x) {
    return (ufs[x] == x)? x : ufs[x] = getf(ufs[x]);
}

struct Edge {
    int u, v, nxt;
} e[MAXN * 460];
int first[MAXN], eCnt;

inline void AddEdge(int w, int u, int v) {
    e[++eCnt].u = u;
    e[eCnt].v = v;
    e[eCnt].nxt = first[w];
    first[w] = eCnt;
}

void solve() {
    read(n);
    for (int i = 1; i <= n; ++i) {
        read(p[i]);
        pos[p[i]] = i;
        ufs[i] = i;
        first[i] = 0;
    }
    eCnt = 0;
    int m = sqrt(n);
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= i + m && j <= n; ++j) {
        	// for index [i,i+sqrt(n)] 
            int tmp = (j - i) * abs(p[j] - p[i]);
            if (tmp < n) {
                AddEdge(tmp, i, j);
            }
            // for permutation [pi,pi+sqrt(n)]
            tmp = (j - i) * abs(pos[j] - pos[i]);
            if (tmp < n) {
                AddEdge(tmp, pos[i], pos[j]);
            }
        }
    }
    // the edges in the e array is already sorted, using union-find set to construct the tree
    LL ans = 0;
    int cnt = n - 1;
    for (int i = 1; i < n; ++i) {
        for (int j = first[i]; j; j = e[j].nxt) {
            int u = getf(e[j].u), v = getf(e[j].v);
            if (u == v) continue;
            ufs[u] = v;
            ans += i;
            --cnt;
        }
        if (cnt == 0) break;
    }
    printf("%lld\n", ans);
}

int main() {
    int T; read(T);
    while (T--) {
        solve();
    }
    return 0;
}
```
<div id="990"></div>
# 1008 [Orgrimmar](http://acm.hdu.edu.cn/showproblem.php?pid=7227)

树形DP：

在树中选取若干个点使点之间最多仅连一条边，最大化点的数量

用 $dp[u][0]$ 表示点 $u$ 不加入集合中的状态，用 $dp[u][1]$ 表示点 $u$ 加入集合，且加入时无结点与其连边（此处可以视作树中线段下端的点），用 $dp[u][2]$ 表示加入时有结点与其连边（线段上端的点），转移方程为
$$
\begin{align}dp[i][0]&={\max_{v\in children[i]}(dp[v][0],dp[v][1],dp[v][2])}\\
dp[i][1]&=\underset{v\in children[i]}{\sum}dp[v][0]\\
dp[i][2]&=\max_{v_0\in children[i]}\left(\left(\underset{v\in children[i],v\ne v_0}{\sum }dp[v][0]\right)+dp[v_0][1],dp[i][2]+dp[v_0][0]\right)
\end{align}
$$
std:

```c++
//1008, std
#include <bits/stdc++.h>
using namespace std;

const int N = 1e6 + 5;

int n, m, x, y, dp[N][3];

vector <int> G[N];

void add(int x, int y) {
	G[x].push_back(y);
	G[y].push_back(x);
}

void dfs(int x, int f) {
	dp[x][0] = 0; dp[x][1] = dp[x][2] = 1;
	for(auto V : G[x]) {
		if(V == f) 
		continue;
		dfs(V, x);
		dp[x][0] += max(dp[V][0], max(dp[V][1], dp[V][2]));
		dp[x][2] += dp[V][0]; 
		dp[x][2] = max(dp[x][2], dp[x][1] + dp[V][1]);
		dp[x][1] += dp[V][0];

	}
}

void rmain() {
	scanf("%d", &n);
//	cerr << n << endl;
	for(int i = 1; i <= n; ++ i) {
		memset(dp[i], 0, sizeof(dp[i]));
		G[i].clear();
	}
	for(int i = 1; i < n; ++ i) {
		scanf("%d%d", &x, &y);
		add(x, y);
	}
	dfs(1, 0);
	cout << max(dp[1][0], max(dp[1][1], dp[1][2])) << endl;
}

int main() {
	freopen("test.txt", "r", stdin);
//	freopen("1008.out", "w", stdout);

int size(512<<20); 
__asm__ ( "movq %0, %%rsp\n"::"r"((char*)malloc(size)+size));
	int T;
	for(cin >> T; T --;) {
		rmain();
	}
	exit(0);
}
```

# 1010 [Vale of Eternal](http://acm.hdu.edu.cn/showproblem.php?pid=7229)

题目大意：$n$ 个点 $(x,y)$ 在经过时间 $t$ 后变为点 $(x\pm t,y)$ 和 $(x,y\pm t)$

$q$ 次询问求所有点构成的多边形面积

![image-20220812140653949](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220812140653949.png)

对于 $t=1$ 的情况我们可以通过求凸包得到面积，但是对于其他询问需要我们在小于 $O(n)$ 的时间复杂度内回答，可以考虑如下做法 $O(1)$ 求出新增面积：

如图，观察得出对于 $t=1$ 之后的每次扩大，斜率绝对值为 $1$ 的边（有且仅有4条）会等差扩大（新增面积为 $|x_{p_1}-x_{p_2}|+1$ ），而其他边会向外平移一个单位（新增面积 $max(|x_{p_1}-x_{p_2}|,|y_{p_1}-y_{p_2}|)$ ），那么对于 $t$ 时刻相比 $t=1$ 时新增加的面积可以表示为 $sum\times(t-1)+2(t-1)^2$ ，即 $n+4$ 个平行四边形面积加上 $8$ 个小三角形面积

> TIPS: 整点多边形的面积为 $S=a+\frac b2-1$，其中 $a$ 为多边形内整点个数， $b$ 为多边形边界上的所有整点个数。对于端点都是整点的线段，其跨过的整点个数为 $gcd(\Delta x,\Delta y)$ 。由此可知整点四边形的面积的2倍是一个整数，可以通过乘2来规避小数运算。

```c++
//1010, std
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e6 + 5;
struct Point {
	int x, y;
	Point(int x = 0, int y = 0) : x(x), y(y) {}
	inline bool operator == (const Point &rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	inline bool operator < (const Point &rhs) const {
		return x == rhs.x ? y < rhs.y : x < rhs.x;
	}
	inline Point operator + (const Point &rhs) const {
		return Point(x + rhs.x, y + rhs.y);
	}
	inline Point operator - (const Point &rhs) const {
		return Point(x - rhs.x, y - rhs.y);
	}
	inline long long operator * (const Point &rhs) const {
		return 1ll * x * rhs.x + 1ll * y * rhs.y;
	}
	inline long long operator ^ (const Point &rhs) const {
		return 1ll * x * rhs.y - 1ll * y * rhs.x;
	}
	inline long long len2() {
		return 1ll * x * x + 1ll * y * y;
	}
}p[maxn];
const Point dir[4] = {Point(0, 1), Point(1, 0), Point(0, -1), Point(-1, 0)};
int n, q;
vector<Point> v;
inline int Left(const Point &a, const Point &b, const Point &c) {
	return ((b - a) ^ (c - a)) >= 0;
}
Point base;
inline vector<Point> Convex(vector<Point>a) {
	int n = a.size(), cnt = 0;
	if(n < 3)
		return a;
	base = a[0];
	for(auto p : a)
		if(make_pair(p.x, p.y) < make_pair(base.x, base.y))
			base = p;
	sort(a.begin(), a.end(), [](const Point &a, const Point &b) {
		long long d = ((a - base) ^ (b - base));
		if(d)
			return d > 0;
		return (a - base).len2() < (b - base).len2();
	});
	vector<Point>res;
	for(int i = 0; i < n; ++ i) {
		while(cnt > 1 && Left(res[cnt - 2], a[i], res[cnt - 1]))
			-- cnt, res.pop_back();
		res.push_back(a[i]), ++ cnt;
	}
	int fixed = cnt;
	for(int i = n - 2; ~i; -- i) {
		while(cnt > fixed && Left(res[cnt - 2], a[i], res[cnt - 1]))
			-- cnt, res.pop_back();
		res.push_back(a[i]), ++ cnt;
	}
	res.pop_back();
	return res;
}
long long ori, sum;
int T;
int main() {
	for(scanf("%d", &T); T --; ) {
		scanf("%d%d", &n, &q);
		v.clear();
		for(int i = 1; i <= n; ++ i) {
			scanf("%d%d", &p[i].x, &p[i].y);
			for(int j = 0; j < 4; ++ j)
				v.push_back(p[i] + dir[j]);
		}
		sort(v.begin(), v.end());
		v.erase(unique(v.begin(), v.end()), v.end());
		v = Convex(v);
		v.push_back(v[0]);
		ori = sum = 0;
		for(int i = 0; i + 1 < (int)v.size(); ++ i)
			ori += abs((v[i] - v[0]) ^ (v[i + 1] - v[0]));
		int cnt = 0;
		for(int i = 0; i + 1 < (int)v.size(); ++ i) {
			long long dx = abs(v[i].x - v[i + 1].x), dy = abs(v[i].y - v[i + 1].y);
			sum += max(dx, dy);
		}
		assert(cnt == 4);
		for(int t; q --; ) {
			scanf("%d", &t);
			long long ans = ori + 2ll * sum * (t-1) + 4ll * (t-1) * (t-1);
			printf("%lld.%d\n", ans >> 1, ans & 1 ? 5 : 0);
		}
	}
}
```

# 1011 [Stormwind](http://acm.hdu.edu.cn/showproblem.php?pid=7230)

签到题：将矩形分为若干小矩形，且面积均大于 $k$，求最大切割数

答案 $\max(\lceil \frac mx\rceil+\lceil \frac ny\rceil,\lceil \frac nx\rceil+\lceil \frac my\rceil)$

$x=\lceil\frac kn\rceil,y=\lceil\frac kx\rceil$或 $x=\lceil\frac km\rceil,y=\lceil\frac kx\rceil$

```c++
#include <iostream>
using namespace std;
void solve(){
	int n,m,k;
	cin>>n>>m>>k;
	if(n>m)swap(n,m);
	int ans1,ans2;
	int x,y;
	x=(k+n-1)/n;
	y=(k+x-1)/x;
	ans1=(m/x-1)+(n/y-1);
	swap(n,m);
	x=(k+n-1)/n;
	y=(k+x-1)/x;
	ans2=(m/x-1)+(n/y-1);
	cout<<max(ans1,ans2)<<endl;
}
signed main(){
//	ios::sync_with_stdio(false),cin.tie(0);
//#ifndef ONLINE_JUDGE
//	freopen("test.txt","r",stdin);
//#endif
	int TT;
//	TT=1;
	cin>>TT;
	while(TT--){
		solve();
	}
	return 0;
}
```