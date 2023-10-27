---
title: '2022 Summer ACM training-Newcoder Vol.10'
date: 2022-08-26
permalink: /posts/2022/08/2022SummerTraining21-Newcoder10/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

![image-20220823115816087](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220823115816087.png)

# B [Fall Guys-Perfect Match](https://ac.nowcoder.com/acm/contest/33195/B)

题目大意： 给一个矩形， 找到一个点， 使得它到矩形中每个数字的最小距离的最大值尽量小。
• 二分答案。
• 将坐标$(x,y)$转化为$(x+y,x-y)$ , 即将曼哈顿距离$|x|+|y|$转为切比雪夫距离$\max(|x|,|y|)$ 。
• 假设二分的答案为$x$， 矩形中可以在$x$的距离内到达某数字$w$的区域可以看成若干个矩形的并。
• 由于对于一个出现$k$次的数字， 将$k$个对应的矩形坐标离散化， 矩形并可以用$O(k^2)$的时间求出。
• 通过预处理矩阵前缀和， 可以求出每个点能在$x$距离内到达的数字个数。 我们只需判断是否存在一个点能到达全部$m$个数字即可。

std:

```c++
#include<bits/stdc++.h>
#define F first
#define S second
#define pb push_back
using namespace std;
typedef pair<int,int>pi;
typedef vector<int>vi;
int n,m,a[1005][1005],b[2005][2005];
vector<pi> G[1000005];
struct node{
    pi p;int v;
};
vector<node> t;
void ADD(int x,int y,int r){
    int lx,ly,rx,ry;
    lx=max(1,x-r); rx=min(2*n+1,x+r+1);
    ly=max(1,y-r); ry=min(2*n+1,y+r+1);
    t.pb((node){(pi){lx,ly},1});
    t.pb((node){(pi){lx,ry},-1});
    t.pb((node){(pi){rx,ly},-1});
    t.pb((node){(pi){rx,ry},1});
}
int ss=0;
int idx[2005],idy[2005];
int sum[45][45],f[45][45];
vector<int> x,y;
void solve(){
    x.clear(); y.clear();
    for (auto nd:t){
        x.pb(nd.p.F);
        y.pb(nd.p.S);
    }
    sort(x.begin(),x.end());
    sort(y.begin(),y.end());
    int p=unique(x.begin(),x.end())-x.begin();
    int q=unique(y.begin(),y.end())-y.begin();
    for (int i=0;i<p;i++) idx[x[i]]=i+1;
    for (int i=0;i<q;i++) idy[y[i]]=i+1;
    for (int i=1;i<=p;i++) for (int j=1;j<=q;j++) sum[i][j]=0;
    for (auto nd:t){
        int xx=idx[nd.p.F],yy=idy[nd.p.S];
        sum[xx][yy]+=nd.v;
    }
    for (int i=1;i<=p;i++)
    for (int j=1;j<=q;j++){
        sum[i][j]+=sum[i][j-1]+sum[i-1][j]-sum[i-1][j-1];
        f[i][j]=(sum[i][j]>0);
    }
    for (int i=0;i<p;i++)
    for (int j=0;j<q;j++){
        int V=f[i][j]-f[i+1][j]-f[i][j+1]+f[i+1][j+1];
        if (V) b[x[i]][y[j]]+=V;
    }
}
bool check(int r){
    memset(b,0,sizeof(b));
    for (int i=1;i<=m;i++){
        t.clear();
        for (auto v:G[i]) ADD(v.F,v.S,r);
        solve();
    }
    for (int i=1;i<=n*2;i++)
    for (int j=1;j<=n*2;j++){
        b[i][j]+=b[i-1][j]+b[i][j-1]-b[i-1][j-1];
    }
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
            if (b[i+j][i-j+n]==m) return 1;
    return 0;
}
int main(){
    ios::sync_with_stdio(false);
    cin >> n >> m;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++){
            cin >> a[i][j];
            G[a[i][j]].pb((pi){i+j,i-j+n});
        }
    for (int i=1;i<=m;i++){
        assert(G[i].size()>=1&&G[i].size()<=20);
    }
    int L=-1,R=n;
    while (R-L>1){
        int mid=(L+R)>>1;
        if (check(mid)) R=mid; else L=mid;
    }
    cout << R << endl;
}
```

# E [Reviewer Assignment](https://ac.nowcoder.com/acm/contest/33195/E)

题目大意：有 $m$ 篇论文和 $n$ 个审稿人,给出每个审稿人能审论文的集合， 要求给每个审稿人安排一篇论文。 令  $f(i)$ 表示被至少i个审稿人审过的论文数量， 要求求出一种分配方案， 使得 $(f(1),f(2),...,f(n))$ 字典序最大

注意到要求f(1)最大就是求二分图最大匹配， 可以用最大流解决。 原问题可以对于 $i=1-n$ 在不改变 $f(1),f(2),...f(i-1)$ 的情况下最大化 $f(i)$， 可以通过做 $n$ 次最大流得到， 每次在上一次最大流结果的基础上建图。

 也可直接建如下最小费用流图:源点向每个审稿人连接容量为 $1$,花费为 $0$ 的边,每个审稿人向能审的论文连接容量为 $1$ ,花费为 $0$ 的边,每篇论文向汇点连接 $n$条容量为 $1$ ,花费为 $1,2,...,n$ 的边， 这里花费的选择也不唯一,只要使得每篇论文被审 $1,2,3...,n$ 次的花费是递增严格凸函数即可。

std:

```c++
#include<bits/stdc++.h>
#define MAXV 805
#define INF 1000000000
using namespace std;
typedef pair<int,int> P;
struct edge{int to,cap,cost,rev;};
int dist[MAXV],h[MAXV],prevv[MAXV],preve[MAXV];
int n,m,V,ans[MAXV];
vector<edge> G[MAXV];
void add_edge(int from,int to,int cap,int cost)
{
    G[from].push_back((edge){to,cap,cost,(int)G[to].size()});
    G[to].push_back((edge){from,0,-cost,(int)G[from].size()-1});
}
int min_cost_flow(int s,int t,int f)
{
    int res=0;
    fill(h+1,h+V+1,0);
    while(f>0)
    {
        priority_queue<P,vector<P>,greater<P> >que;
        fill(dist+1,dist+V+1,INF);
        dist[s]=0;
        que.push(P(0,s));
        while(!que.empty())
        {
            P p=que.top(); que.pop();
            int v=p.second;
            if(dist[v]<p.first) continue;
            for(int i=0;i<G[v].size();i++)
            {
                edge &e=G[v][i];
                if(e.cap>0&&dist[e.to]>dist[v]+e.cost+h[v]-h[e.to])
                {
                    dist[e.to]=dist[v]+e.cost+h[v]-h[e.to];
                    prevv[e.to]=v;
                    preve[e.to]=i;
                    que.push(P(dist[e.to],e.to));
                }
            }
        }
        if(dist[t]==INF)
        {
            return -1;
        }
        for(int v=1;v<=V;v++) h[v]+=dist[v];
        int d=f;
        for(int v=t;v!=s;v=prevv[v])
        {
            d=min(d,G[prevv[v]][preve[v]].cap);
        }
        f-=d;
        res+=d*h[t];
        for(int v=t;v!=s;v=prevv[v])
        {
            edge &e=G[prevv[v]][preve[v]];
            e.cap-=d;
            G[v][e.rev].cap+=d;
        }
    }
    return res;
}
int main()
{
    scanf("%d%d",&n,&m);
    V=n+m+2;
    int s=V-1,t=V;
    for(int i=1;i<=n;i++) add_edge(s,i,1,0);
    for(int i=1;i<=m;i++)
        for(int j=1;j<=n;j++)
            add_edge(n+i,t,1,j);
    for(int i=1;i<=n;i++)
    {
        string str;
        cin>>str;
        for(int j=0;j<m;j++)
            if(str[j]=='1') 
                add_edge(i,n+j+1,1,0);
    }
    int res=min_cost_flow(s,t,n);
    if(res==-1) puts("-1");
    else
    {
        for(int i=1;i<=n;i++)
            for(auto e:G[i])
            {
                if(e.to>=n+1&&e.to<=n+m&&(!e.cap)) {ans[i]=e.to-n; break;}
            }
        for(int i=1;i<=n;i++)
        {
            assert(ans[i]>=1&&ans[i]<=m);
            printf("%d%c",ans[i],i==n?'\n':' ');
        }
    }
    return 0;
}
```

# F [Shannon Switching Game?](https://ac.nowcoder.com/acm/contest/33195/F)

题目大意： 给定一个无向图， 初始时有一个token在s点， 两个玩家Join Player和Cut Player轮流行动， Cut Player先动。 Cut Player每次可以移除一条和token所在位置相邻的边, Join Player每次可以将token沿着一条未删除边移动, 如果token在某刻被移动到t则Join Player获胜， 否则Cut Player获胜， 求双方最优策略下的胜者。

我们可以求出在双方最优策略下使得Join Player可以将token移动到t的所有token起点集合,把这个集合叫做good set
• 首先,t点一定是在good set中的， 我们从t开始逐步构建good set。 某个不在good set中的顶点v如果有至少两条边连向good set中的某个顶点， 那么从该点出发的话， 由于Cut Player在一次操作中只能切断其中的一条边， 那么Join Player一定可以在一次操作后将token移动到good set中的某个顶点， 因此此时v也在good set中。
• 如果在某一时刻， 任何不在good set中的顶点都只有至多一条边连向good set中的某个顶点,那么从不在good set中的任一顶点出发， Cut Player只需要每次切断可能连向good set的边即可,那么此时不在good
set中的顶点一定都不能到达t点。
• 构建可以通过维护一个队列来实现， 时间复杂度为O(n+m),其中n是顶点个数,m是边数 。

赛时将题目误读为可以删去所有边

std:

```c++
#include<bits/stdc++.h>
#define pb push_back
using namespace std;
const int maxn=1e6+10;
vector<int> G[maxn];
int n,m,d[maxn],s,t;
void solve(){
    cin >> n >> m >> s >> t; assert(s!=t);
    for (int i=1;i<=n;i++) d[i]=0,G[i].clear();
    for (int i=0;i<m;i++){
        int u,v; cin >> u >> v;
        G[u].pb(v); G[v].pb(u);
    }
    queue<int> q; q.push(t); d[t]=2;
    while (!q.empty()){
        int u=q.front(); q.pop();
        for (auto v:G[u]){
            d[v]++;
            if (d[v]==2) q.push(v);
        }
    }
    if (d[s]<2) puts("Cut Player"); else puts("Join Player");
}
int main(){
    ios::sync_with_stdio(false);
    int _; cin >> _;
    while (_--) solve();
}
```

# H [Wheel of Fortune](https://ac.nowcoder.com/acm/contest/33195/H)

题目大意：

炉石：两人卡牌游戏，$A,B$每方拥有一张英雄卡与不多于7张随从卡，触发了某种机制等可能随机对其中一张卡牌造成10点伤害，给出卡牌生命值，求 $A$ 获胜概率

易知答案与随从的生命值无关

设英雄卡血量为 $A,B$ , 设 $x=\lceil \frac{A}{10}\rceil$, $y=\lceil \frac{B}{10}\rceil$  

则答案为
$$
\sum_{i=0}^{x-1}\mathbb{C}_{i+y-1}^i\times2^{-(y+i)}
$$
相当于将 $A,B$ 的血量拆成 $x$ 个黑球与 $y$ 个白球，对于前 $i+y-1$ 个球（拿出一个白球并固定放在最后一个）黑白球混合的情况总数为 $\mathbb{C}_{i+y-1}^i$（可通过通用求组合数算法推出），每种情况的可能性为 $(2^i)^{-1}$ 

或按照通用求组合数算法：

对前 $i$ 个球进行排列，每种情况的可能性为所有球全排列 $i!$ 除以黑球全排列 $(i-y)!$ 后，减去前一方案数，最后乘$(2^i)^{-1}$ ，即
$$
f(i)=\frac{i!}{(i-y)!}-f(i-1)&f(y)=1,i=y+1,y+2,\dots,y+x-1\\
ans=\sum_{i=y
}^{y+x-1} f(i)\times(2^i)^{-1}
$$


```c++
#include <bits
#define rep(_it,_lb,_ub) for(int _it=(_lb);_it<=(_ub);_it++)
#define rep2(_it,_ub,_lb) for(int _it=(_ub);_it>=(_lb);_it--)
using namespace std;
typedef long long ll;
typedef long double ld;
//#define double ld
//#define int ll
inline char gc(){static char buf[100000],*p1=buf,*p2=buf;return p1==p2&&(p2=(p1=buf)+fread(buf,1,100000,stdin),p1==p2)?EOF:*p1++;}
#define gc getchar
//inline ll read(){char c=gc();ll su=0,f=1;for (;c<'0'||c>'9';c=gc()) if (c=='-') f=-1;for (;c>='0'&&c<='9';c=gc()) su=su*10+c-'0';return su*f;}
template <typename T>inline void read(T &s){s=0;T w = 1;char c=gc();while(c<'0'||c>'9'){if(c=='-')w=-1;c=gc();}while(c>='0'&&c<='9'){s=(s<<1)+(s<<3)+c-'0';c=gc();}s*=w;}
const ll mod=998244353;
const ll maxn=100005;
const ll inf=0x3f3f3f3f;
const double pi=acos(-1);
typedef pair<int,int> pii;
ll fact[2300000];
ll inv[2300000];	//inv(2^i)
ll Inv[2300000];	//inv(i)
ll qpow(ll a,ll x){
    ll ans=1,tmp=a;
    while(x){
        if(x&1)ans=tmp*ans%mod;
        tmp=tmp*tmp%mod;
        x>>=1;
    }
    return ans;
}
void init(ll N){
	fact[0]=fact[1]=1;
	inv[1]=qpow(2,mod-2);
    Inv[1]=1;
	for(ll i=2;i<=N;i++){
		fact[i]=fact[i-1]*i%mod;
		inv[i]=qpow(qpow(2,i),mod-2);
        Inv[i]=(mod-mod/i)*Inv[mod%i]%mod;
	}
}
ll f[1000005];
ll cal(ll bx,ll k){
	if(k==0)return f[0]=1,inv[bx+k];
	ll ans=(fact[bx+k]*qpow(fact[bx],mod-2)%mod*qpow(fact[k],mod-2)%mod-f[k-1]+mod)%mod;
	f[k]=(f[k-1]+ans)%mod;
	return ans*inv[bx+k]%mod;
}
void solve(){
	ll a,b,c;
    cin>>a;for(int i=0;i<7;i++) cin>>c;
    cin>>b;for(int i=0;i<7;i++) cin>>c;
    a=(a+9)/10,b=(b+9)/10;
    init(2*max(a,b)+5);
    ll ret=0;
    for(ll i=0;i<=a-1;i++){
    	ll tmp=cal(b,i);
//    	cout<<tmp<<endl;
        ret=(ret+tmp)%mod;
	}
	cout<<ret<<endl;
}
signed main(){
//	ios::sync_with_stdio(false),cin.tie(0);
//#ifndef ONLINE_JUDGE
//	freopen("test.txt","r",stdin);
//#endif
	int TT;
	TT=1;
//	read(TT);
	while(TT--){
		solve();
	}
	return 0;
}
```

# I [Yet Another FFT Problem?](https://ac.nowcoder.com/acm/contest/33195/I)

给出长度为$n$的数组$A$，长度为$m$的数组B，验证是否存在 $i,j,k,l$ 使
$$
1\le i\ne j\le n,1\le k\ne l\le m;\\
|a_i-a_j|=|b_k-b_l|
$$
并输出一组解

不失一般性，假设A,B中无重复元素，题目要求等价于
$$
1\le i\ne j\le n,1\le k\ne l\le m;\\
|a_i-a_j|=|b_k-b_l|
$$
遍历序列$\{a_i+b_j\}$，由抽屉原理，仅需要遍历 $O(V)$ 次便能求出一组解或判断不存在，其中 $V=\max(\max\{a_i\},\max\{b_i\})$

std:

```c++
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define MAXN 1000005
#define MAXM 10000005
#define INF 1000000000
#define MOD 1000000007
#define F first
#define S second
using namespace std;
typedef long long ll;
typedef pair<int,int> P;
int n,m,k,a[MAXN],b[MAXN];
P save[2*MAXM];
int pa[MAXM],pb[MAXM];
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    for(int i=1;i<=m;i++) scanf("%d",&b[i]);
    P p=P(0,0),q=P(0,0);
    vector<int> va,vb;
    memset(pa,0,sizeof(pa));
    memset(pb,0,sizeof(pb));
    for(int i=1;i<=n;i++)
        if(pa[a[i]]) p=P(pa[a[i]],i); else {pa[a[i]]=i; va.push_back(i);}
    for(int i=1;i<=m;i++)
        if(pb[b[i]]) q=P(pb[b[i]],i); else {pb[b[i]]=i; vb.push_back(i);}
    if(p.F!=0&&q.F!=0)
    {
        printf("%d %d %d %d\n",p.F,p.S,q.F,q.S);
        return 0;
    }
    for(int i=1;i<=20000000;i++) save[i]=P(0,0);
    for(int i=0;i<(int)va.size();i++)
        for(int j=0;j<(int)vb.size();j++)
        {
            int sum=a[va[i]]+b[vb[j]];
            if(save[sum].F)
            {
                printf("%d %d %d %d\n",save[sum].F,va[i],min(vb[j],save[sum].S),max(vb[j],save[sum].S));
                return 0;
            }
            save[sum]=P(va[i],vb[j]);
        }
    puts("-1");
    return 0;
}
```