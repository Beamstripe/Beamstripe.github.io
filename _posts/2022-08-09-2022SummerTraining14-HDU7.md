---
title: '2022 Summer ACM training-HDU Vol.7'
date: 2022-08-09
permalink: /posts/2022/08/2022SummerTraining14-HDU7/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

![image-20220809190513690](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220809190513690.png)

# 1002 [Independent Feedback Vertex Set](http://acm.hdu.edu.cn/showproblem.php?pid=7210)

题目大意：将无向图的所有结点分为两个集合，一个是独立集 $V_I$ ，另一个 $V_F$ 的导出子图形成森林（无环）。求 $V_I$ 的最大点权和。

> 独立集：图的一一个顶点子集称为独立集，如果该子集中的任意两个项点在图中不相邻。图 G 的最大独立集所包含顶点的个数称作 G 的独立数(independence number), 记作 $\alpha(G)$

\*该图的构建方式是从 $(1,2), (1,3), (2,3)$ 开始每次选取图中已连边的两点 $u,v$ 将 $(u,x),(v,x)$ 加入图中

由此图的构建方式可知对于任意一个三元环，有且必只有一个能加入独立集中，若无则不满足森林的定义，若两个及以上则不满足独立集的定义

那么可以将问题转化为图的三染色问题，对应相同颜色的就是独立集的解（恰有三个），求最值即可

std:

```c++
#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
int main() {
    int T; scanf("%d", &T);
    while (T--) {
        int n; scanf("%d", &n);
        vector<int> w(n);
        vector<int> c(n, 0);
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        for (int i = 0; i < n; ++i)
            scanf("%d", &w[i]);
        for (int i = 3; i < n; ++i) {
            int j, k;
            scanf("%d %d", &j, &k);
            j--; k--;
            c[i] = (3 - c[j] - c[k]) % 3;
        }
        long long ans[3] = { 0, 0, 0 };
        for (int i = 0; i < n; ++i)
            ans[c[i]] += w[i];
        cout << max({ ans[0], ans[1], ans[2] }) << endl;
    }
    return 0;
}
```

# 1003 [Counting Stickmen](http://acm.hdu.edu.cn/showproblem.php?pid=7211)

题目大意：在图中找出火柴人的个数，火柴人如图

![stickman-graph-example](http://acm.hdu.edu.cn/data/images/C1050-1003-1.png)

可以通过寻找身体上端结点（头的下端，图中3）和身体下端结点（腿的上端，图中5）来定位火柴人。

因此枚举所有边进行计数，假设身体靠头一侧的端点为 $x$ , 靠腿一侧的端点为 $y$,我们在 $x$ 处计算头和手的方案, 在 $y$ 处计算腿的方案, 根据乘法原理, 相乘即可得到火柴人的种数.

设结点 $i$ 出边条数（双向）为 $deg_i$ ，则腿的数量为 $\mathbb{C}_{deg_y-1}^2$ 

选取手外侧结点：个数为$handnode=(\sum_{u\in neighbor(x)}deg_u)-deg_x-(deg_y-1)$ 方案有 $\mathbb{C}^2_{handnode}$

即所有子结点出边个数去掉回连的 $deg_x$ 与不应计算在内的腿部分 $deg_y-1$ (不一定是腿边)

而手臂有公共结点的方案是 $\sum_{u\in neighbor(x),u\ne y}\mathbb{C}_{deg_u-1}^2$ 

则手臂总方案数为 $\mathbb{C}^2_{handnode}-\sum_{u\in neighbor(x),u\ne y}\mathbb{C}_{deg_u-1}^2$

选头的方案数为 $deg_x-3$ ，相乘得到答案

需要 $O(n)$ 预处理 $deg_x,\sum_{u\in neighbor(x)}(deg_u-1),\sum_{u\in neighbor(x)}\mathbb{C}_{deg_u-1}^2$

之后 $O(n)$ 遍历所有边即可

std:

```c++
#include<bits/stdc++.h>

using namespace std;

#define gc c=getchar()
#define r(x) read(x)
#define ll long long

template<typename T>
inline void read(T &x){
    x=0;T k=1;char gc;
    while(!isdigit(c)){if(c=='-')k=-1;gc;}
    while(isdigit(c)){x=x*10+c-'0';gc;}x*=k;
}

const int N=1e7+7;
const int p=998244353;

vector<int>G[N];
int deg[N];
int s0[N];
int s1[N];
int s2[N];

inline int add(int a,int b){
    return (a+=b)>=p?a-p:a;
}

inline int sub(int a,int b){
    return (a-=b)<0?a+p:a;
}

inline ll calc(int x){
    return (ll)x*(x-1)/2%p;
}

int main(){
    // freopen(".in","r",stdin);
    // freopen(".out","w",stdout);
    int T;
    r(T);
    while(T--){
    int n;r(n);
    for(int i=1;i<=n;++i)G[i].clear(),deg[i]=s0[i]=s1[i]=s2[i]=0;
    for(int i=1;i<n;++i){
        int u,v;r(u),r(v);
        G[u].push_back(v);
        G[v].push_back(u);
        ++deg[u],++deg[v];
    }
    for(int x=1;x<=n;++x){
        s0[x]=deg[x];
        for(auto &y:G[x]){
            s1[x]=add(s1[x],deg[y]-1);
            s2[x]=add(s2[x],calc(deg[y]-1));
        }
    }
    int ans=0;
    for(int x=1;x<=n;++x){
        if(s0[x]>=4){
            for(auto &y:G[x]){
                if(s0[y]>=3){
                    int head=sub(s0[x],3);
                    int foot=sub(s0[y],1);
                    int hand=sub(s1[x],foot);
                    ans=add(ans,(ll)head*sub(calc(hand),sub(s2[x],calc(deg[y]-1)))%p*calc(foot)%p);
                }
            }
        }
    }
    printf("%d\n",ans);
    }
    return 0;
}
```

# 1004 [Black Magic](http://acm.hdu.edu.cn/showproblem.php?pid=7212)

签到模拟：

```c++
#include <iostream>
#include <algorithm>
using namespace std;
//int a[3];
void solve(){
	int a,b,c,d;
	int ans;
	scanf("%d%d%d%d",&a,&b,&c,&d);
	if(d==0) {
		ans=a+b+c;
	}
	else {
		if((b+c)>0) ans=a+b+c;
		else ans=a+b+c+1;
	}
	//cout<<ans<<endl;
	ans=ans-min(c,b);
	cout<<ans<<" ";
	ans=b+c;
	if(a==0){
		if(d==0) cout<<ans<<endl;
		else cout<<ans+1<<endl;
	}
	else {
		int s=a;
	    s=s+2;
		//cout<<ans<<endl;
	    ans+=min(d,s-1);
	    cout<<ans+a<<endl;
	}
}
int main(){
	int t;
	cin>>t;
	while(t--){
		solve();
	}
	return 0;
}
```

# 1005 [Sumire](http://acm.hdu.edu.cn/showproblem.php?pid=7214)

数位DP: std不太好改天补题

std:

```c++
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define se second
#define SZ(x) ((int)x.size())
#define lowbit(x) x&-x
#define pb push_back
#define ALL(x) (x).begin(),(x).end()
#define UNI(x) sort(ALL(x)),x.resize(unique(ALL(x))-x.begin())
#define GETPOS(c,x) (lower_bound(ALL(c),x)-c.begin())
#define LEN(x) strlen(x)
#define MS0(x) memset((x),0,sizeof((x)))
#define Rint register int
#define ls (u<<1)
#define rs (u<<1|1)
typedef unsigned int unit;
typedef long long ll;
typedef unsigned long long ull;
typedef double db;
typedef pair<int,int> pii;
typedef pair<ll,ll> pll;
typedef vector<int> Vi;
typedef vector<ll> Vll;
typedef vector<pii> Vpii;
template<class T> void _R(T &x) { cin >> x; }
void _R(int &x) { scanf("%d", &x); }
void _R(ll &x) { scanf("%lld", &x); }
void _R(ull &x) { scanf("%llu", &x); }
void _R(double &x) { scanf("%lf", &x); }
void _R(char &x) { scanf(" %c", &x); }
void _R(char *x) { scanf("%s", x); }
void R() {}
template<class T, class... U> void R(T &head, U &... tail) { _R(head); R(tail...); }
template<class T> void _W(const T &x) { cout << x; }
void _W(const int &x) { printf("%d", x); }
void _W(const ll &x) { printf("%lld", x); }
void _W(const double &x) { printf("%.16f", x); }
void _W(const char &x) { putchar(x); }
void _W(const char *x) { printf("%s", x); }
template<class T,class U> void _W(const pair<T,U> &x) {_W(x.fi);putchar(' '); _W(x.se);}
template<class T> void _W(const vector<T> &x) { for (auto i = x.begin(); i != x.end(); _W(*i++)) if (i != x.cbegin()) putchar(' '); }
void W() {}
template<class T, class... U> void W(const T &head, const U &... tail) { _W(head); putchar(sizeof...(tail) ? ' ' : '\n'); W(tail...); }
const int MOD=1e9+7,mod=998244353;
ll qpow(ll a,ll b) {ll res=1;a%=MOD; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%MOD;a=a*a%MOD;}return res;}
const int MAXN=5e5+10,MAXM=1e7+10;
const int INF=INT_MAX,SINF=0x3f3f3f3f;
const ll llINF=LLONG_MAX;
const int inv2=(MOD+1)/2;
const int Lim=1<<20;

template <int _P>
struct Modint
{
    static constexpr int P=_P;
private :
    int v;
public :
    Modint() : v(0){}
    Modint(ll _v){v=_v%P;if(v<0)v+=P;}
    explicit operator int() const {return v;}
    explicit operator long long() const {return v;}
    explicit operator bool() const {return v>0;}
    bool operator == (const Modint &o) const {return v==o.v;}
    bool operator != (const Modint &o) const {return v!=o.v;}
    Modint operator - () const {return Modint(v?P-v:0);}
    Modint operator + () const {return *this;}
    Modint & operator ++ (){v++;if(v==P)v=0;return *this;}
    Modint & operator -- (){if(v==0)v=P;v--;return *this;}
    Modint operator ++ (int){Modint r=*this;++*this;return r;}
    Modint operator -- (int){Modint r=*this;--*this;return r;}
    Modint & operator += (const Modint &o){v+=o.v;if(v>=P)v-=P;return *this;}
    Modint operator + (const Modint & o)const{return Modint(*this)+=o;}
    Modint & operator -= (const Modint & o){v-=o.v;if(v<0)v+=P;return *this;}
    Modint operator - (const Modint &o)const {return Modint(*this)-=o;}
    Modint & operator *=(const Modint & o){v=(int)(((ll)v)*o.v%P);return *this;}
    Modint operator * (const Modint & o)const {return Modint(*this)*=o;}
    Modint & operator /= (const Modint & o){return (*this)*=o.Inv();}
    Modint operator / (const Modint & o)const{return Modint(*this)/=o;}
    friend Modint operator + (const Modint &x,const ll &o) {return x+(Modint)o;}
    friend Modint operator + (const ll &o,const Modint &x) {return x+(Modint)o;}
    friend Modint operator - (const Modint &x,const ll &o) {return x-(Modint)o;}
    friend Modint operator - (const ll &o,const Modint &x) {return (Modint)o-x;}
    friend Modint operator * (const Modint &x,const ll &o) {return x*(Modint)o;}
    friend Modint operator * (const ll &o,const Modint &x) {return x*(Modint)o;}
    friend Modint operator / (const Modint &x,const ll &o) {Modint c=o;return x*c.Inv();}
    friend Modint operator / (const ll &o,const Modint &x) {Modint c=o;return c*x.Inv();}
    Modint operator ^ (ll o)const{Modint r=1,t=v;while(o){if(o&1)r*=t;t*=t;o>>=1;}return r;}
    Modint operator ~ (){return (*this)^(P-2);}
    Modint Inv() const{return (*this)^(P-2);}
};

using mi=Modint<MOD>;

template<int P>
void _W(Modint<P> x){printf("%d",(int)x);}

template<int P>
void _R(Modint<P> &x){ll t;scanf("%lld",&t);x=t;}

mi dp[75][75][2][2],vis[75][75][2][2];;
ll t;
int s[75],k,b,d,n,m,c;

mi dfs(int dep,int tot,int lim,bool zero)
{
    if(dep==m+1&&tot==0)return 1;
    if(dep==m+1)return 0;
    if(tot<0)return 0;
    if(vis[dep][tot][lim][zero])return dp[dep][tot][lim][zero];
    vis[dep][tot][lim][zero]=1;
    int up=lim?s[dep]:b-1;
    int ct=0,i=0;
    int c=(i==d);
    if(zero&&(d==0))c=0;
    dp[dep][tot][lim][zero]+=dfs(dep+1,tot-c,lim&&(s[dep]==i),zero);
    ct++;
    if(i!=d&&d<=up)
    {
        ct++;
        i=d;
        int c=(i==d);
        if(zero&&(d==0))c=0;
        dp[dep][tot][lim][zero]+=dfs(dep+1,tot-c,lim&&(s[dep]==i),0);
    }
    if(i!=up)
    {
        ct++;
        i=up;
        dp[dep][tot][lim][zero]+=dfs(dep+1,tot,lim&&(s[dep]==i),zero&&(i==0));
    }
    dp[dep][tot][lim][zero]+=dfs(dep+1,tot,0,0)*max(0,up-ct+1);
    return dp[dep][tot][lim][zero];
}


mi calc(bool f)
{
    MS0(dp);MS0(vis);
    R(t);
    m=0;
    t-=f;
    while(t)
    {
        s[++m]=t%b;
        t/=b;
    }
    reverse(s+1,s+m+1);
    mi ans=0;
    for(int i=1;i<=m;i++)
    {
        mi t=i;
        ans+=dfs(1,i,1,1)*(t^k);
    }
    return ans;
}

void solve()
{
    R(k,b,d);
    mi ans=calc(1);
    ans=calc(0)-ans;
    W(ans);
}

int main()
{
    srand(time(0));
    int T=1;
    scanf("%d",&T);
    for(int kase=1;kase<=T;kase++)
    {
        //printf("Case #%d: ",kase);
        solve();
    }
    return 0;
}
```

# 1006 [Triangle Game](http://acm.hdu.edu.cn/showproblem.php?pid=7216)

博弈论：

一个非退化三角形，三边边长分别为$a,b,c$ 。现 Kate 和 Emilico 二人做游戏，每轮需要令三角形的一边长度
减去一正整数，使这个三角形退化的一方负。Kate 先手，双方均采用最优策略，问 Kate 是否会获胜。
结论是：Kate 获胜当且仅当 $(a-1)\oplus(b-1)\oplus(c-1)\ne0$ 。其中 为异或运算。
不妨令 $x,y,z$ 分别为三角形最短，次短和最长的边长。由于 $x+y>z$ 且 $x,y,z$都是正整数，则有 $(x-1)+(y-1)\ge(z-1)$。
不妨定义如下状态：
（L 态）$(x-1)\oplus(y-1)\oplus(z-1)=0$
（W 态）$(x-1)\oplus(y-1)\oplus(z-1)\ne0$
当玩家目前处于 L 态时，由于$(x-1)\oplus(y-1)=(z-1)$，则有 $(x-1)+(y-1)\ge(x-1)\oplus (y-1)=(z-1)$，此时一定为一个非退化三角形。
在 L 态时，转移有如下情况：
$x=1$，则 $y=z$ 。这种情况任何玩家移动都会判负，因此为必败态。
 $x>1$，则 $1<x<y<z$ ，此时一定存在一种减少边长的方案，使得减少边长后三角形不退化。不妨
 $x$ 考虑 减少为 $x^\prime$ ，则由于减去的是正整数，会有 $(y-1)\oplus(z-1)=(x-1)\ne(x^\prime-1)$。则 $(x^\prime-1)\oplus(y-1)\oplus(z-1)\ne0$
。如果改变的是其他边，也可以类似地利用此式。
综上，L 态一定转移到某必败态或 W 态。
当玩家处于 W 态时，有 $(x-1)\oplus(y-1)\ne(z-1)$。令 $r=(x-1)\oplus(y-1)\oplus(z-1)$，则 $((x-1)\oplus r)+1<x$
或 $((y-1)\oplus r)+1<y$ 或 $((z-1)\oplus r)+1<z$，三式中必有一式成立。考
虑 $r$ 的二进制中每一位 $1$ 都一定出现了奇数次，对于最高位的 $1$，将其异或 $r$ 后一定会变小。因此可以将成
立的不等式作为这一步操作进行代换，即可转为 L 态。
由此，每次移动三角形某边长均会减小，W 态会转化为 L 态，而 L 态均转化为某些必败态和 W 态。由此，所
有 W 态均为必胜态，L 态均为必败态。结论证毕