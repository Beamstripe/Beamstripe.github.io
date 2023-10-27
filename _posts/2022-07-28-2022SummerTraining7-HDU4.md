---
title: '2022 Summer ACM training-HDU Vol.4'
date: 2022-07-28
permalink: /posts/2022/07/hd4/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# 1001 [Link with Bracket Sequence II](http://acm.hdu.edu.cn/showproblem.php?pid=7174)

题目大意：一个未完全填充的字符串（长度 $n\le500$ ），求按照括号序列的要求填充序列情况总数（可能有 $10^9+7$ 种不同的括号）

区间DP:设 $f_{i,j}$ 表示 为合法括号序列且 $i,j$ 上括号相互匹配的方案数， $g_{i,j}$ 表示 $i,j$ 区间形成一个合法括号序列的方案数，转移为：

$$
f_{l,r}=e\times g_{l+1,r-1}\\
g_{l,r}=g_{l,r}+\sum_{i=l}^rg_{l,i-1}\times f_{i,r}
$$

其中
$$
e=\begin{cases}
m,&a_l=a_r=0\\
1,&a_l+a_r=0\wedge a_l>0\wedge a_r<0 \vee a_l\times a_r=0\\
0,&others
\end{cases}
$$

标程：

```c++
#include <cstdio>
#include <cstring>
#define MN 500

using ll = long long;

const int mod = 1000000007;

int n,m;
int a[MN+5];
int f[MN+5][MN+5],g[MN+5][MN+5];

void solve(){
    memset(f,0,sizeof(f));
    memset(g,0,sizeof(g));
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
    }
    if(n&1){
        puts("0");
        return;
    }
    for(int i=0;i<=n;i++){
        g[i+1][i] = 1;
    }
    for(int len=2;len<=n;len+=2){
        for(int l=1;l+len-1<=n;l++){
            int r = l+len-1;
            if(a[l]>=0&&a[r]<=0){
                int e;
                if(a[l]==0&&a[r]==0){
                    e = m;
                }else if(a[l]==0||a[r]==0){
                    e = 1;
                }else if(a[l]+a[r]==0){
                    e = 1;
                }else{
                    e = 0;
                }
                f[l][r] = (ll)g[l+1][r-1]*e%mod;
            }
            for(int nl=l;nl<=r;nl+=2){
                g[l][r] = (g[l][r]+(ll)g[l][nl-1]*f[nl][r])%mod;
            }
        }
    }
    printf("%d\n",g[1][n]);
}

int main(){
	freopen("test.in",stdin);
    int T;
    scanf("%d",&T);
    //T = 1;
    while(T--) solve();
}
```

# 1002 [Link with Running](http://acm.hdu.edu.cn/showproblem.php?pid=7175)

题目大意：最短路图上路径最大权值和（满足最短路情况下权值最大）（路径长可能为0）

考虑在最短路图上跑最长路

无非正环的最短路图必是DAG，但**此题会出现零环(\*环边长度均为0)**，需要缩点成DAG

实际上题目数据**保证零环边的权值均为0**，满足题目一定存在答案的要求

```c++
#include <cstdio>
#include <algorithm>
#include <functional>
#include <stack>
#include <queue>
#include <vector>
#define MN 100000
 
using std::min;
using std::max;
using std::function;
using std::greater;
using std::swap;
using std::stack;
using std::queue;
using std::priority_queue;
using std::vector;
 
using ll = long long;
 
const ll INF = 1e18;
 
struct Edge{
    int v,w1,w2;
};
 
namespace GetSpg{
    ll dis[MN+5];
    vector<Edge> e[MN+5];
    
    void clear(int n){
        for(int i=1;i<=n;i++){
            e[i].clear();
        }
    }
    
    void addEdge(int u,int v,int w1,int w2){
        e[u].push_back({v,w1,w2});
    }
    
    void dijkstra(int n,int S){
        using pii = std::pair<ll,int>;
        priority_queue<pii,vector<pii>,greater<pii>> pq;
        for(int i=1;i<=n;i++){
            dis[i] = INF;
        }
        pq.push({dis[S]=0,S});
        while(!pq.empty()){
            int u = pq.top().second;
            ll d = pq.top().first;
            pq.pop();
            if(d!=dis[u]) continue;
            for(Edge edge:e[u]){
                int v = edge.v;
                int w = edge.w1;
                if(dis[u]+w<dis[v]){
                    dis[v] = dis[u]+w;
                    pq.push({dis[v],v});
                }
            }
        }
    }
    
    void solve(int n,function<void(int,int,int,int)> addEdge){
        dijkstra(n,1);
        for(int u=1;u<=n;u++){
            if(dis[u]==INF) continue;
            for(Edge edge:e[u]){
                if(dis[u]+edge.w1==dis[edge.v]){
                    addEdge(u,edge.v,edge.w1,edge.w2);
                }
            }
        }
    }
    
}
 
namespace GetDag{
    vector<Edge> e[MN+5];
    
    stack<int> s;
    bool ins[MN+5];
    int low[MN+5],dfn[MN+5],scc[MN+5];
    int dfnCnt=0,sccCnt=0;
    
    void clear(int n){
        for(int i=1;i<=n;i++){
            e[i].clear();
            ins[i] = false;
            dfn[i] = low[i] = scc[i] = 0;
        }
        dfnCnt = 0;
        sccCnt = 0;
        while(!s.empty()) s.pop();
    }
    
    void addEdge(int u,int v,int w1,int w2){
        e[u].push_back({v,w1,w2});
    }
    
    void tarjan(int u){
        dfn[u] = ++dfnCnt;
        low[u] = dfn[u];
        s.push(u);
        ins[u] = true;
        for(Edge edge:e[u]){
            int v = edge.v;
            if(dfn[v]){
                if(ins[v]){
                    low[u] = min(low[u],dfn[v]);
                }
            }else{
                tarjan(v);
                low[u] = min(low[u],low[v]);
            }
        }
        if(low[u]==dfn[u]){
            int v;
            ++sccCnt;
            do{
                v = s.top();
                s.pop();
                ins[v] = false;
                scc[v] = sccCnt;
            }while(u!=v);
        }
    }
    
    void solve(int& n,function<void(int,int,int,int)> addEdge,bool isLoop[]){
        for(int i=1;i<=n;i++){
            if(!dfn[i]){
                tarjan(i);
            }
        }
        for(int u=1;u<=n;u++){
            for(Edge edge:e[u]){
                int v = edge.v;
                if(scc[u]==scc[v]){
                    if(edge.w2>0){
                        isLoop[scc[u]] = true;
                    }
                }else{
                    addEdge(scc[u],scc[edge.v],edge.w1,edge.w2);
                }
            }
        }
    }
    
}
 
namespace GetLp{
    int din[MN+5];
    bool isLoop[MN+5];
    vector<Edge> e[MN+5];
    
    struct Dis{
        ll d;
        Dis(ll d=0){
            this->d = d;
        }
        Dis operator + (const Dis& that)const{
            if(d==-INF||that.d==-INF) return Dis(-INF);
            if(d==INF||that.d==INF) return Dis(INF);
            return Dis(d+that.d);
        }
        bool operator < (const Dis& that)const{
            return this->d < that.d;
        }
    };
    
    Dis f[MN+5];
    
    void clear(int n){
        for(int i=1;i<=n;i++){
            din[i] = 0;
            isLoop[i] = false;
            e[i].clear();
        }
    }
    
    void addEdge(int u,int v,int w1,int w2){
        e[u].push_back({v,w1,w2});
        din[v]++;
    }
 
    void solve(int n,int S){
        for(int i=1;i<=n;i++){
            f[i] = -INF;
        }
        f[S] = 0;
        queue<int> q;
        for(int i=1;i<=n;i++){
            if(din[i]==0) q.push(i);
        }
        while(!q.empty()){
            int u = q.front();
            q.pop();
            if(isLoop[u]) f[u] = f[u]+INF;
            for(Edge edge:e[u]){
                int v = edge.v;
                int w = edge.w2;
                f[v] = max(f[v],f[u]+w);
                if(--din[v]==0){
                    q.push(v);
                }
            }
        }
    }
}
 
void solve(){
	
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++){
        int u,v,w1,w2;
        scanf("%d%d%d%d",&u,&v,&w1,&w2);
        GetSpg::addEdge(u,v,w1,w2);
    }
    GetSpg::solve(n,GetDag::addEdge);
    GetDag::solve(n,GetLp::addEdge,GetLp::isLoop);
    GetLp::solve(GetDag::sccCnt,GetDag::scc[1]);
    printf("%lld %lld\n",GetSpg::dis[n],GetLp::f[GetDag::scc[n]].d);
    GetSpg::clear(n);
    GetDag::clear(n);
    GetLp::clear(n);
}
 
int main(){
//	freopen("test.txt","r",stdin); 
//	freopen("test.txt","w",stdout); 
    int T;
    scanf("%d",&T);
    while(T--) solve();
}
```



# 1004 [Link with Equilateral Triangle](http://acm.hdu.edu.cn/showproblem.php?pid=7177)

题目大意：按照如下规则填充0,1,2且满足所有小三角形顶点和均不为3的倍数

![](http://acm.hdu.edu.cn/data/images/C1047-1004-1.png)

答案：对任意 $n$ 不存在方案满足条件

对于一个合法的解，应当满足不存在同时包含0,1,2的三角形，下面我们证明这样的三角形一定存在。
左下角必然是1，右下角必然是0，底边不能含有2，则底边上必然有奇数条1-0的边，这些边都属于一个
小三角形。考虑其他的0-1边，由于不在两个斜边上，其他的0-1边必然属于两个三角形。因此“每个三角
形内0-1边的数量”的和必然为奇数。
但是，假设不存在0-1-2的三角形，则所有三角形都必然包含0条或2条的0-1边，产生了矛盾。
因此一定存在0-1-2的三角形。

# 1006 [BIT Subway](http://acm.hdu.edu.cn/showproblem.php?pid=7179)

题目大意：买 $n$ 张票，满100元后之后的票价打8折，满200后之后的票价打5折，求恰好凑折扣买票的价格与正常买票的价格

e.g. 在花了199元后买10元与8元车票：

凑折扣： $199￥+1.25*0.8￥+8.75*0.5￥+10*0.5￥=208.375￥$ 

正常：

 $199￥+10*0.8￥+8*0.5￥=211￥$ 

易知对于方案一：
$$
ans=\begin{cases}
x&0\le x\le 100\\
(x-100)*0.8+100&100\le x<225\\
(x-225)*0.5+200&x\ge225
\end{cases}
$$
赛时代码：

```c++
#include <queue>
#include <set>
#include <iostream>
#include <cstdio>
#include <map>
#include <bitset>
#include <cmath>
#include <stack>
#include <ctime>
#include <list>
#include <cassert>
#include <functional>
#include <iomanip>
#include <complex>
#include <algorithm>
#include <cstring>
#include <string>
#include <cmath>
#define N 10009
#define MOD 998244353
#define int long long
using namespace std;
typedef long long ll;
typedef long double ld;
inline char gc(){static char buf[100000],*p1=buf,*p2=buf;return p1==p2&&(p2=(p1=buf)+fread(buf,1,100000,stdin),p1==p2)?EOF:*p1++;}
#define gc getchar
inline ll read(){char c=gc();ll su=0,f=1;for (;c<'0'||c>'9';c=gc()) if (c=='-') f=-1;for (;c>='0'&&c<='9';c=gc()) su=su*10+c-'0';return su*f;}
//ios_base::sync_with_stdio(0);
const ll inf=1000000000000000LL;
const int mod=1e9+7;;
const int maxn=200009;
const double pai=3.1415926;
int a[maxn];
double ans=0,anss=0,num=1,numm=1;
void solve(){
	int n;
	cin>>n;
	ans=0,anss=0,num=1,numm=1;
	for(int i=0;i<n;i++){
		cin>>a[i];
		//cout<<ans<<endl;
		ans=ans+a[i]*num;
		//cout<<ans<<" "<<a[i]<<" "<<num<<endl;
		anss=anss+a[i]*numm;
		if(ans==100){
			num=0.8;
		}
		else if(ans==200){
			if(num==1){
				double xx=ans-a[i]*num;
				if(a[i]>=125+100-xx){
					xx=a[i]-100+xx-125;
					ans=200+xx*(0.5);num=0.5;
				}
				else{
					xx=a[i]-(100-xx);
					ans=100+xx*(0.8);num=0.8;
				}
			}
			else{
				num=0.5;
			}
		}
		else if(ans>200){
			double xx=ans-a[i]*num;
			if(xx<100){
				if(a[i]>=125+100-xx){
					xx=a[i]-100+xx-125;
					ans=200+xx*(0.5);num=0.5;
				}
				else{
					xx=a[i]-(100-xx);
					ans=100+xx*(0.8);num=0.8;
				}
			}
			else if(xx<200&&xx>=100){
				if(a[i]>=(200-xx)/(num)){
					xx=a[i]-(200-xx)/(num);
					ans=200+xx*(0.5);num=0.5;
				}
				else{
					ans=xx+a[i]*(0.8);num=0.8;
				}
			}
		}
		else if(ans>100&&ans<200){
			double xx=ans-a[i]*num;
			if(xx<100){
				ans=ans-a[i];
				double yy=100-xx;
				yy=yy/(num);num=0.8;
				xx=a[i]-yy;
				ans=100+(xx*num);
			}
		}
		if(anss>=100&&anss<200){
			numm=0.8;
		}
		else if(anss>=200){
			numm=0.5;
		}
	}
	cout<<fixed<<setprecision(3)<<ans<<' '<<anss<<endl;
	return;
}




signed main(){
	ios_base::sync_with_stdio(0);cin.tie(0);
	int t;
	cin>>t;
	while(t--){
		solve();
	}
	return 0;
} 

```

之后应注意先简化式子后编程

# 1007 [Climb Stairs](http://acm.hdu.edu.cn/showproblem.php?pid=7180)

题目大意：

有 $n$ 级台阶，每个台阶都有一个BOSS血量 $a_i$ ，一个人从0层开始，每次可以向上跳最多k个台阶或向下跳1个台阶（不可回到已击败BOSS的台阶），并击败该层台阶的BOSS，前提是攻击力大于其血量，每次攻击后攻击力提升值为该层BOSS的血量值，给出初始攻击力，问是否能打败所有BOSS

```c++
#include <iostream>
#include<cmath>
#define ll long long
using namespace std;
ll a[500005];
int main(){
	int t;cin>>t;
	while(t--){
		ll n,a0,k;cin>>n>>a0>>k;
		ll f=0,m=0;	
		//ll s=0;
		ll sk=k;
		if(k>n) k=n;
		for(int i=0;i<n;i++) scanf("%d",&a[i]);
		for(int i=0;i<n;i++){
		//	cout<<i<<endl;
			if(a[i]<=a0) {
				a0=a0+a[i];
				k=sk; 
			}
			else {
			//	cout<<a0<<endl;
				ll m=a[i],j=0,s=0;
				while((a[i+j]+a0)<m||a[i+j]>a0){
					//cout<<a[i+j]<<" "<<a0<<" "<<m<<endl;
					if(i+j>=n+k) break;
					m=max(a[i+j],abs(a[i+j]-m));
					s=s+a[i+j];
					j++;
				}
			//	cout<<a[i+j]<<" "<<a0<<" "<<m<<endl;
				if(a[i+j]+a0<m) {
					f=1;
					break;
				}
				//cout<<i<<' '<<j<<" "<<k<<endl;
				if(j>=k||i+j>=n) 	{
					f=1;
					break;
				}
				a0=a0+s+a[i+j];
				k=sk-j;
				i=i+j; 
			}
			//cout<<a0<<endl;
		}
		if(f==0) cout<<"YES"<<endl;
		else cout<<"NO"<<endl;
	}
} 
```

贪心，按照要先到达的右端点 $r$ ，依次维护区间 $[l,r]$ ，其中 $r$ 的查找需要满足最小

标程：

```c++
#include<bits/stdc++.h>
#define rep(i,s,t) for(int i=(s),i##end=(t);i<=i##end;++i)
#define dwn(i,s,t) for(int i=(s),i##end=(t);i>=i##end;--i)
#define ren for(int i=fst[x];i;i=nxt[i])
#define Fill(a,x) memset(a,x,sizeof(a))
using namespace std;
typedef long long ll;
typedef double db;
typedef long double ld;
typedef unsigned long long ull;
typedef vector<int> vi;
typedef pair<int,int> pii;
const int inf=2139062143;
const int MOD=998244353;
const int MAXN=200100;
inline int read()
{
    int x=0,f=1;char ch=getchar();
    while(!isdigit(ch)) {if(ch=='-') f=-1;ch=getchar();}
    while(isdigit(ch)) {x=x*10+ch-'0';ch=getchar();}
    return x*f;
}
inline ll readll()
{
    ll x=0,f=1;char ch=getchar();
    while(!isdigit(ch)) {if(ch=='-') f=-1;ch=getchar();}
    while(isdigit(ch)) {x=x*10+ch-'0';ch=getchar();}
    return x*f;
}
namespace CALC
{
    inline int pls(int a,int b){return a+b>=MOD?a+b-MOD:a+b;}
    inline int mns(int a,int b){return a-b<0?a-b+MOD:a-b;}
    inline int mul(int a,int b){return (1LL*a*b)%MOD;}
    inline void inc(int &a,int b){a=pls(a,b);}
    inline void dec(int &a,int b){a=mns(a,b);}
    inline void tms(int &a,int b){a=mul(a,b);}
    inline int qp(int x,int t,int res=1)
        {for(;t;t>>=1,x=mul(x,x)) if(t&1) res=mul(res,x);return res;}
    inline int Inv(int x){return qp(x,MOD-2);}
}
using namespace CALC;
int n,a[MAXN],k,las,cur,l,r;
pair<ll,int> q[MAXN];
ll sum[MAXN],f[MAXN],mxf[MAXN];
ll mn[MAXN<<2];
void build(int k,int l,int r)
{
    if(l==r) {mn[k]=f[l];return ;}int mid=l+r>>1;
    build(k<<1,l,mid);build(k<<1|1,mid+1,r);
    mn[k]=min(mn[k<<1],mn[k<<1|1]);
}
int res=inf;
void query(int k,int l,int r,int a,int b,int w)
{
    if(a<=l&&r<=b)
    {
        if(mn[k]>w||res!=inf) return ;
        if(l==r) {res=l;return ;}int mid=l+r>>1;
        if(mn[k<<1]<=w) query(k<<1,l,mid,a,b,w);
        else if(mn[k<<1|1]<=w) query(k<<1|1,mid+1,r,a,b,w);
        return ;
    }
    int mid=l+r>>1;
    if(a<=mid) query(k<<1,l,mid,a,b,w);
    if(b>mid) query(k<<1|1,mid+1,r,a,b,w);
}
int solve()
{
    n=read(),a[0]=sum[0]=read(),k=min(read(),n);
    rep(i,1,n) a[i]=read(),sum[i]=sum[i-1]+a[i];
    f[0]=sum[0]<<1;rep(i,1,n) f[i]=max(sum[i]+a[i],f[i-1]);
    rep(i,0,n) f[i]=f[i]-sum[i];
    //rep(i,0,n) cout<<f[i]<<" ";puts("");
    build(1,1,n);
    las=-1,cur=0;
    while(cur!=n)
    {
        res=inf;
        query(1,1,n,cur+1,min(las+k+1,n),sum[cur]);
        //cout<<las<<" "<<cur<<" "<<res<<endl;
        if(res==inf) return 0;
        las=cur,cur=res;
    }
    return 1;
}
int main()
{
    rep(T,1,read()) puts(solve()?"YES":"NO");
}
```



# 1011 [Link is as bear](http://acm.hdu.edu.cn/showproblem.php?pid=7184)

一个序列，每次修改 $[l,r]$ 内的所有数 $a_l,a_{l+1},\dots a_{r}$ 为区间抑或和 $s=a_l\oplus a_{l+1}\dots\oplus a_r$，最后要处理成一个所有数相等的序列，问这个数的最大值

问题完全等价于给n个数，从中选一些数，使得这些数的异或和最大。即为线性基的板子

赛时代码：

```
#include <iostream>
#include <cstring>
using namespace std;
typedef long long ll;
const int MAXN = 100005;
ll d[64];
void add(ll x)
{
    for(int i=60;i>=0;i--)
    {
        if(x&(1ll<<i))
        {
            if(d[i])x^=d[i];
            else
            {
                d[i]=x;
                break;
            }
        }
    }
}
ll ans()
{
    ll anss=0;
    for(int i=60;i>=0;i--)
    if((anss^d[i])>anss)anss^=d[i];
    return anss;
}  
void solve(){
	memset(d,0,sizeof(d));
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		ll x;
		cin>>x;
		add(x);
	}
	cout<<ans()<<'\n';
}
int main(){
	ios::sync_with_stdio(false),cin.tie(0);
	int t;
	cin>>t;
	while(t--)solve();
	return 0;
}
```