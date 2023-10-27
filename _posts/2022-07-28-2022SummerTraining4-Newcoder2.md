---
title: '2022 Summer ACM training-Newcoder Vol.2'
date: 2022-07-28
permalink: /posts/2022/07/nc2/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# D [Link with Game Glitch](https://ac.nowcoder.com/acm/contest/33187/D)

题目大意：交换物品可以使$a_i$个物品$b_i$，变为$w\times c_i$个物品$d_i$，求不能无限交换的临界最大值$w$

考虑建立一个边权为$\frac{c_i}{a_i}$ 的有向图

(Bellman-Ford算法）每次最短路松弛时乘以$w$，之后反向松弛（dis置0，每次松弛使边权变长）

(SPFA算法)每次计算距离远近的方法为 $dis*w^{cur}$ ,其中 $cur$ 为目前已松弛的点数，算法思想类似判负环（若如此松弛一个点超过n次，则可以无限松弛，即无限兑换物品）

可以尝试将所有值取对数来避免大数乘法计算

std: SPFA

```c++
#include <cstdio>
#include <cmath>
#include <queue>
#include <vector>
#define MN 1000

using std::log;
using std::queue;
using std::vector;

using ld = long double;

struct Edge{
	int v;
	ld w;
};

int n,m;
int din[MN+5];
vector<Edge> e[MN+5];
ld psw;

void addEdge(int u,int v,ld w){
	e[u].push_back({v,w});
	din[v]++;
}

struct Dis{
	ld dis;
	int cnt;
	
	void reset(){
		dis = 0;
		cnt = 0;
	}
	
	void setInf(){
		dis = 1e100;
		cnt = 0;
	}
	
	bool operator < (const Dis& that)const{
		return dis+cnt*psw < that.dis+that.cnt*psw;
	}
	
	Dis operator + (ld w)const{
		return {dis+w,cnt+1};
	}
	
};

Dis dis[MN+5];
bool inq[MN+5];

bool hasNegativeLoop(){
	queue<int> q;
	for(int i=1;i<=n;i++){
		dis[i].reset();
		q.push(i);
		inq[i] = true;
	}
	while(!q.empty()){
		int u = q.front();
		q.pop();
		inq[u] = false;
		if(dis[u].cnt>=n) return true;
		for(Edge edge:e[u]){
			int v = edge.v;
			ld w = edge.w;
			if(dis[u]+w<dis[v]){
				dis[v] = dis[u]+w;
				if(!inq[v]){
					q.push(v);
					inq[v] = true;
				}
			}
		}
	}
	return false;
}

bool check(ld psw){
	::psw = psw;
	return !hasNegativeLoop();
}

int main(){
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++){
		int a,u,c,v;
		scanf("%d%d%d%d",&a,&u,&c,&v);
		ld w = -log((ld)c/a);
		addEdge(u,v,w);
	}
	ld l=0,r=1;
	for(int t=0;t<60;t++){
		ld mid = (l+r)/2;
		if(check(-log(mid))){
			l = mid;
		}else{
			r = mid;
		}
	}
	printf("%.10f\n",(double)r);
}
```



# E [Falfa with Substring](https://ac.nowcoder.com/acm/contest/33187/E)

题目大意：一个长度为 $n$ 的字符串中恰含 $k$ 个"bit"的所有小写字母字符串个数为 $F_{n,k}$，给定 $n$ 求
$$
(F_{n,k})_{k=0}^{n}
$$
先求至少出现k次bit时字符串的个数

设一个字符串中"bit"为一个点，将其他无关字符视作一个点，易求得字符串个数为$F_k=\frac{(n-2k)!}{(n-3k)!k!}\times 26^{n-3k}$ 或 $\begin{pmatrix}n-2k\\k\end{pmatrix}\times 26^{n-3k}$

利用容斥原理，列出答案$G_k=\sum_{j\ge k}\begin{pmatrix}j\\k\end{pmatrix}(-1)^{j-k}F_j$

对表达式变形得：$k!G_k=\sum_{j\ge k}(j!F_j)\frac{(-1)^{j-k}}{(j-k)!}$

设 $P_i=i!F_i$,  $Q_i=\frac{(-1)^{j-k}}{(j-k)!}$,  $R_{n+i}=i!G_i$，用NTT加速计算该式卷积即可

std: 带NTT卷积模板

```c++
#include<bits/stdc++.h>

using ll = long long;
using ld = long double;

namespace GTI
{
	char gc(void)
   	{
		const int S = 1 << 16;
		static char buf[S], *s = buf, *t = buf;
		if (s == t) t = buf + fread(s = buf, 1, S, stdin);
		if (s == t) return EOF;
		return *s++;
	}
	ll gti(void)
   	{
		ll a = 0, b = 1, c = gc();
		for (; !isdigit(c); c = gc()) b ^= (c == '-');
		for (; isdigit(c); c = gc()) a = a * 10 + c - '0';
		return b ? a : -a;
	}
	int gts(char *s)
   	{
		int len = 0, c = gc();
		for (; isspace(c); c = gc());
		for (; c != EOF && !isspace(c); c = gc()) s[len++] = c;
		s[len] = 0;
		return len;
	}
	int gtl(char *s)
   	{
		int len = 0, c = gc();
		for (; isspace(c); c = gc());
		for (; c != EOF && c != '\n'; c = gc()) s[len++] = c;
		s[len] = 0;
		return len;
	}
}
using GTI::gti;
using GTI::gts;
using GTI::gtl;

const int N = 1 << 20 | 1, M = 998244353, G = 3;
int qpw(int a, int b)
{
	a %= M;
	if (a == 0) return (b == 0);
	b %= M - 1;
	if (b < 0) b += M - 1;
	int c = 1;
	for (; b; b >>= 1, a = 1ll * a * a % M)
		if (b & 1)
			c = 1ll * c * a % M;
	return c;
}
int mod(int val)
{
	return (val < 0) ? (val + M) : ((val >= M) ? (val - M) : val);
}

namespace Poly
{
	std::vector<int> g[2][20];
	int getgl(int len, int tag)
	{
		return qpw(G, (M - 1) / len * tag);
	}
	void precalc(void)
	{
		for (int l = 1, x = 0; l < (1 << 20); l <<= 1, ++x)
		{
			int gl[2] = {getgl(l << 1, 1), getgl(l << 1, -1)}, gx[2] = {1, 1};
			g[0][x].resize(l), g[1][x].resize(l);
			for (int i = 0; i < l; i++)
				for (int t = 0; t < 2; t++)
				{
					g[t][x][i] = gx[t];
					gx[t] = 1ll * gx[t] * gl[t] % M;
				}
		}
	}
	int id[N];
	int init(int n)
	{
		int k = 0, len = 1;
		while (len < n) len <<= 1, ++k;
		for (int i = 0; i < len; i++)
			id[i] = id[i >> 1] >> 1 | ((i & 1) << (k - 1));
		return len;
	}
	void ireverse(int *a, int len)
	{
		for (int i = 0; i < len; i++)
			if (id[i] > i)
				std::swap(a[id[i]], a[i]);
	}
	void ntt(int *a, int len, int tag = 1)
	{
		tag = (tag < 0) ? 1 : 0;
		ireverse(a, len);
		for (int l = 1, x = 0; l < len; l <<= 1, ++x)
			for (int st = 0; st < len; st += (l << 1))
				for (int i = st; i < st + l; i++)
				{
					int tmp = 1ll * a[i + l] * g[tag][x][i - st] % M;
					a[i + l] = mod(a[i] - tmp);
					a[i] = mod(a[i] + tmp);
				}
		if (tag)
		{
			int rev = qpw(len, -1);
			for (int i = 0; i < len; i++)
				a[i] = mod(1ll * a[i] * rev % M);
		}
	}
}

int fct[N], ifc[N];
void init(int n)
{
	fct[0] = 1;
	for (int i = 1; i <= n; i++)
		fct[i] = 1ll * fct[i - 1] * i % M;
	ifc[n] = qpw(fct[n], -1);
	for (int i = n - 1; i >= 0; i--)
		ifc[i] = ifc[i + 1] * (i + 1ll) % M;
}
int C(int n, int m)
{
	if (m < 0 || n - m < 0) return 0;
	return 1ll * fct[n] * ifc[m] % M * ifc[n - m] % M;
}

int f[N], g[N];
int main(void)
{
	int n = gti(), m = n / 3;
	init(n), Poly::precalc();
	for (int i = 0; i <= m; i++)
		f[i] = 1ll * C(n - i * 2, i) * qpw(26, n - i * 3) % M;
	for (int i = 0, sgn = 1; i <= m; i++, sgn = -sgn)
	{
		f[i] = 1ll * f[i] * fct[i] % M;
		g[m - i] = sgn * ifc[i];
	}

	int len = Poly::init(m * 2 + 1);
	Poly::ntt(f, len), Poly::ntt(g, len);
	for (int i = 0; i < len; i++)
		f[i] = 1ll * f[i] * g[i] % M;
	Poly::ntt(f, len, -1);

	for (int i = 0; i <= m; i++)
		f[i + m] = 1ll * f[i + m] * ifc[i] % M;
	for (int i = 0; i <= m; i++)
		printf("%d%c", f[m + i], " \n"[i == n]);
	for (int i = m + 1; i <= n; i++)
	    printf("%d%c", 0, " \n"[i == n]);
	return 0;
}
```

# 

# G [Link with Monotonic Subsequence](https://ac.nowcoder.com/acm/contest/33187/G)

构造一个排列，使其 $\max(lis(p), lds(p))$ 最小。

首先给出结论：排列权值的最小值为$\lceil\sqrt{n}\rceil$

Dilworth's theorem: 对偏序集 $<A，≤>$ ，设A中最长链的长度是n，则将A中元素分成不相交的反链，反链个数至少是n。

标程：

```c++
#include <cstdio>
#include <cmath>
#define MN 1000000

int n,a[MN+5];

void solve(){
	scanf("%d",&n);
	int B = ceil(sqrt(n));
	int b = n%B;
	for(int i=B;i<=n;i+=B){
		for(int j=1;j<=B;j++){
			a[i-B+j] = i-j+1;
		}
	}
	for(int i=1;i<=b;i++){
		a[n-b+i] = n-i+1;
	}
	for(int i=1;i<=n;i++){
		printf("%d%c",a[i]," \n"[i==n]); 
	}
}

int main(){
	int T;
	scanf("%d",&T);
	while(T--) solve();
}
```

# H [Take the Elevator](https://ac.nowcoder.com/acm/contest/33187/H)

题目大意：n 个人坐电梯，楼高 m ，每个人有起始楼层和目标楼层。
电梯有载客量限制 k ，上升时可以上升到任意层并随时下降，但是下降
要一直下降到一层才能再上升。
电梯每秒运行一层，换方向和上下人不占用时间，问电梯最短运行时间。

贪心：具体来说，我们可以把每个人看成一条线段 $[l, r]$：
如果当前电梯内人数不足 k，只需要找一条还未选择的 r最大且
$r ≤ r_{max}$的线段（起始楼层不能低于已经上了电梯的人）
如果当前电梯内人数为 k，只需要找一条还未选择的 r最大且 $r ≤ l_{max}$的
线段（有人下电梯）
下行其实是一样的，只不过变成贪心选择起始楼层最高的
每轮上行下行取一个 max 即可，最后复杂度是 $O(n \log n)$的

标程：

```c++
#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N=2e5+10;
int n,m,k;
int a[N],b[N];

struct node{
	int pos,id;
	node(int _p=0,int _i=0):pos(_p),id(_i){}
	bool operator <(const node&rhs)const{
		if(pos==rhs.pos) return id<rhs.id;
		return pos>rhs.pos;
	}
};
multiset<node>downa,downb,upa,upb;

int main(){
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;++i){
		scanf("%d%d",&a[i],&b[i]);
		if(a[i]>b[i]) downa.insert(node(a[i],i));
		else upb.insert(node(b[i],i));
	}
	ll ans=0;
	while(n){
		int mx=0;
		auto l=downa.begin(),r=upb.begin();
		if(l==downa.end()) mx=r->pos;
		else if(r==upb.end()) mx=l->pos;
		else mx=max(l->pos,r->pos);
		ans+=(mx-1)*2;

		for(int nowp=mx,cnt=0;;){
			while(cnt<m){
				auto t=upb.lower_bound(node(nowp,0));
				if(t==upb.end()) break;
				++cnt;--n;nowp=t->pos;
				upa.insert(node(a[t->id],t->id));
				upb.erase(t);
			}
			if(!upa.size()) break;
			auto t=upa.begin();
			nowp=t->pos;--cnt;
			upa.erase(t);
		}
		for(int nowp=mx,cnt=0;;){
			while(cnt<m){
				auto t=downa.lower_bound(node(nowp,0));
				if(t==downa.end()) break;
				++cnt;--n;nowp=t->pos;
				downb.insert(node(b[t->id],t->id));
				downa.erase(t);
			}
			if(!downb.size()) break;
			auto t=downb.begin();
			nowp=t->pos;--cnt;
			downb.erase(t);
		}
	}
	printf("%lld\n",ans);
	return 0;
}
/*
5 1 6
1 3
2 4
5 6
5 4
4 2

14
*/
```

# J [Link with Arithmetic Progression](https://ac.nowcoder.com/acm/contest/33187/J)

裸最小二乘法，求残差平方和$\sum_{i=1}^n(y_i-\hat y_i)^2$

精度问题：

$$\hat b=\frac{\sum_{i=1}^n(x_i-\overline x)(y_i-\overline y)}{\sum_{i=1}^n(x_i-\overline x)^2}$$

$$
=\frac{\sum_{i=1}^nx_iy_i-n\overline x\overline y}{\sum_{i=1}^nx_i^2-n\overline x^2}
$$

$$
\hat a=\overline y-\hat b\overline x
$$

在实际求解残差时：

公式(1)**精度较高**，利用double即可

公式(2)精度低，需要用**long double**

后来听说答案用python跑的，只能说少用C++浮点吧

```c++
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cctype>
#define double long double
namespace GTI
{
    char gc(void)
       {
        const int S = 1 << 16;
        static char buf[S], *s = buf, *t = buf;
        if (s == t) t = buf + fread(s = buf, 1, S, stdin);
        if (s == t) return EOF;
        return *s++;
    }
    int gti(void)
       {
        int a = 0, b = 1, c = gc();
        for (; !isdigit(c); c = gc()) b ^= (c == '-');
        for (; isdigit(c); c = gc()) a = a * 10 + c - '0';
        return b ? a : -a;
    }
}
using GTI::gti;
using namespace std;
const int MAXN = 1000016;
double a[MAXN];
double s[MAXN];
double p[MAXN];
typedef long long ll;
inline char gc(){static char buf[100000],*p1=buf,*p2=buf;return p1==p2&&(p2=(p1=buf)+fread(buf,1,100000,stdin),p1==p2)?EOF:*p1++;}
#define gc getchar
inline ll read(){char c=gc();ll su=0,f=1;for (;c<'0'||c>'9';c=gc()) if (c=='-') f=-1;for (;c>='0'&&c<='9';c=gc()) su=su*10+c-'0';return su*f;}
//ios_base::sync_with_stdio(0);
void solve(){
	ll n=gti();
	double xis=1.0*n*(n+1);
    xis*=(2*n+1)/6.0;
    double xi=1.0*n*(n+1)/2.0;
    a[0]=p[0]=s[0]=0;
	for(ll i=1;i<=n;i++){
		a[i]=gti();
		p[i]=p[i-1]+i*a[i];
		s[i]=s[i-1]+a[i];
	}
	double k=(n*p[n]-xi*s[n])/(n*xis-xi*xi);
	double b=s[n]/n-k*(n+1)/2.0;
	double res=0;
	for(ll i=1;i<=n;i++){
		res+=(k*i+b-a[i])*(k*i+b-a[i]);
	}
	cout<<fixed<<setprecision(12)<<res<<endl;
	
} 
int main(){
	int T;
	cin>>T;
	while(T--)solve();
	return 0;
}
```

# K [Link with Bracket Sequence I](https://ac.nowcoder.com/acm/contest/33187/K)

题目大意：

已知括号序列 $a$ 是一个长度为 $m$ 的合法括号序列 $b$ 的子序列，求可能
的序列 $b$ 的数量。

实际上这道题与下面这道题很像：

[括号序列-蓝桥杯](https://blog.csdn.net/qq_45302640/article/details/122726632?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165855011816780357256759%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165855011816780357256759&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122726632-null-null.142^v33^experiment_28w_v1,185^v2^control&utm_term=%E6%8B%AC%E5%8F%B7%E5%BA%8F%E5%88%97&spm=1018.2226.3001.4187)

记 dpi,j,k 表示在序列 b 的前 i 位中，与 a 的 lcs 为 j ，且左括号比右括
号多 k 个的方案数。
转移时枚举下一位填写的是哪种括号即可。

```c++
#include <cstdio>
#include <cstring>
#define MN 200

const int mod = 1000000007;

int n,m;
char s[MN+5]; 

int f[MN+5][MN+5][MN+5];

void add(int& a,int b){
	a = (a+b)%mod;
}

void solve(){
	scanf("%d%d%s",&n,&m,&s[1]);
	memset(f,0,sizeof(f[0])*(m+1));
	f[0][0][0] = 1;
	for(int i=0;i<m;i++){
		for(int j=0;j<=i;j++){
			for(int k=0;k<=i;k++){
				if(j>0){
					if(s[k+1]==')'){
						add(f[i+1][j-1][k+1],f[i][j][k]);
					}else{
						add(f[i+1][j-1][k],f[i][j][k]);
					}
				} 
				{
					if(s[k+1]=='('){
						add(f[i+1][j+1][k+1],f[i][j][k]);
					}else{
						add(f[i+1][j+1][k],f[i][j][k]);
					}
				}
			}
		}
	}
	printf("%d\n",f[m][0][n]);
}

int main(){
	int T;
	scanf("%d",&T);
	while(T--) solve();
}
```