---
title: '2022 Summer ACM training-HDU Vol.9'
date: 2022-08-16
permalink: /posts/2022/08/hd9/
tags:
  - Chinese post
  - ACM
  - Algorithm
---
![image-20220816194306642](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220816194306642.png)

# 1001[Arithmetic Subsequence](http://acm.hdu.edu.cn/showproblem.php?pid=7233)

题目描述. 给一个长度为$N$的整数序列 $A=(A_i)$，问能否将该序列重排得到序列 $B=(B_i)$，
满足$1\le i<j<k\le N$ ; $(B_i,B_j,B_k)$ 不构成等差序列（$N\le5000$）。

首先如果某个数出现次数大于等于3则不存在解（其他情况必有解）。然后如果所有数字均为偶数，我们可将所有数除以二;如果所有数字均为奇数，我们可将所有数减去一;否则,我们将所有奇数放在左边，所有偶数放在右边，对奇数/偶数分治解决。单组数据时间复杂度$O(n \log \max A_i)$。

std将所有数按二进制位颠倒后进行排序，效果相同

```c++
// #pragma comment(linker, "/STACK:102400000,102400000")
#pragma GCC optimize("O3")
#pragma GCC optimize("O2")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")
// #pragma GCC optimize("trapv")
#include<bits/stdc++.h>
// #include <bits/extc++.h>
#define int long long
#define double long double
// #define i128 long long
// #define double long double
using namespace std;
 
#define rep(i,n) for (int i=0;i<(int)(n);++i)
#define rep1(i,n) for (int i=1;i<=(int)(n);++i)
#define range(x) begin(x), end(x)
#define sz(x) (int)(x).size()
#define pb push_back
#define F first
#define S second
 
typedef long long ll;
typedef unsigned long long ull;
// typedef long double ld;
typedef pair<int, int> pii;
typedef vector<int> vi;
 
int dx[]={1,-1,0,0};
int dy[]={0,0,1,-1};
const int mod=998244353;
const double EPS=1e-9;
const double pi=acos(-1);
const int INF=1e18;
const int N=5007;
mt19937 rng(1235);
int n;
vi ans;
void solve(vi &x){
    vector<pair<int,int>> y;
    int n=sz(x);
    for (int i=0;i<n;++i){
        y.push_back({0,x[i]});
        for (int j=0;j<30;++j){
            if (x[i]>>j&1) y[i].first|=(1ll<<(29-j));
        }
    }
    sort(range(y));
    for (int i=0;i<n;++i) ans.push_back(y[i].second);
}
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cout.precision(15);
    int _;
    cin>>_;
    while (_--){
	    cin>>n;
	    vi a(n,0);
	    rep(i,n) cin>>a[i];
	    map<int,int> cnt;
	    ans.clear();
	    rep(i,n){
		    cnt[a[i]]++;
		    if (cnt[a[i]]>2){cout<<"NO\n"; goto cont;} 
	    }
	    cout<<"YES\n";
	    solve(a);
	    for (auto c:ans) cout<<c<<" ";
	    cout<<"\n";
	    cont:;
    }
	return 0;
}
```

# 1003 [Fast Bubble Sort](http://acm.hdu.edu.cn/showproblem.php?pid=7235)

题目大意：

给定任何一个长度为$N$的数组$A=(a_1,a_2,\dots,a_n)$,令$B(A)$表示对$A$进行一次bubble
sort循环之后得到的数组。令$num(A)$表示从$A$到$B(A)$最少需要移动元素(数组区间循环移位)的次
数。给定一个$1-N$的排列$P$以及$q$组$1\le l\le r\le N$，求$num(P_{[l,r]})$

易知题目与区间局部最大值的个数有关

假设$P=n_1\lambda_1n_2\lambda_2\dots n_k\lambda_k$,则$B(P)=\lambda_1n_1\lambda_2n_2\dots \lambda_k n_k$,其中$n_1,\dots,n_k$为从左到右的局部最大值且有$n_1 < n_2 < \dots< n_k$,则不难证明答案为非空 $\lambda_i$ 的个数。

将询问离线,每次从n到1倒序扫描左端点????并回答所有左端点为????的询问。对于每个固定的左端点 $l$，$[l,n]$ 中从左到右的局部最大值可以通过单调栈维护，局部最大值插入/删除对于答案的影响可以用树状数组/线段树快速更新/求解（在遍历到局部最大值时将原先的剔除。单组数据时间复杂度为 $O((n+q)\log n)$

```c++
// #pragma comment(linker, "/STACK:102400000,102400000")
#pragma GCC optimize("O3")
#pragma GCC optimize("O2")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")
// #pragma GCC optimize("trapv")
#include<bits/stdc++.h>
// #include <bits/extc++.h>
#define int long long
#define double long double
// #define i128 long long
// #define double long double
using namespace std;
 
#define rep(i,n) for (int i=0;i<(int)(n);++i)
#define rep1(i,n) for (int i=1;i<=(int)(n);++i)
#define range(x) begin(x), end(x)
#define sz(x) (int)(x).size()
#define pb push_back
#define F first
#define S second
 
typedef long long ll;
typedef unsigned long long ull;
// typedef long double ld;
typedef pair<int, int> pii;
typedef vector<int> vi;


namespace internal {


int ceil_pow2(int n) {
    int x = 0;
    while ((1U << x) < (unsigned int)(n)) x++;
    return x;
}

int bsf(unsigned int n) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
#else
    return __builtin_ctz(n);
#endif
}

}  // namespace internal

using S=int;
S op(S l,S r){return l+r;}
S e(){return 0;}


int dx[]={1,-1,0,0};
int dy[]={0,0,1,-1};
const int mod=998244353;
const int base[]={12321,32123};
const double EPS=1e-9;
const double pi=acos(-1);
const int INF=1e18;
const int N=100017;
mt19937 rng(1235);

int n,q;
int p[N];
int ans[N],C[N],now[N];
vector<pii> info[N];
int st[N];

int lowbit(int u){return u&(-u);}
void update(int u,int w){for (int i=u;i<=n+7;i+=lowbit(i)) C[i]+=w;}
int query(int u){int ans=0; for (int i=u;i;i-=lowbit(i)) ans+=C[i]; return ans;}
signed main(){
  ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
  int _;
  cin>>_;
  //_=1;
  while (_--){
    cin>>n>>q;
    rep(i,n) cin>>p[i];
    for (int i=1;i<=n+5;++i) C[i]=0,now[i]=0;
    rep(i,N) info[i].clear();
    p[n]=n+1;
    rep(i,q){
      int u,v;
      cin>>u>>v;
      u--, v--;
      info[u].pb({i,v});
    }
    int cnt=0;
    st[cnt++]=n; 
    auto upd=[&](int u){
      update(u+1,(now[u+1]?-1:1)), now[u+1]^=1;
      update(u+2,(now[u+2]?-1:1)), now[u+2]^=1;
    };
    for (int i=n-1;i>-1;--i){
      while (cnt&&p[i]>p[st[cnt-1]]) {
        cnt--; upd(st[cnt]);
      }
      st[cnt++]=i, upd(i);
      for (auto c:info[i]){
        int id=c.F, r=c.S;
        if (i==r) ans[id]=0;
        else ans[id]=(query(r+1)-query(i))/2;
      }
    }
    rep(i,q) cout<<ans[i]<<"\n";
  }
  return 0;
}
```



# 1007 [Matryoshka Doll](http://acm.hdu.edu.cn/showproblem.php?pid=7239)

$dp(x,y)$ 表示将前????个套娃分成????组的方案数，转移为
$dp(x,y) = dp(x-1,y-1) + dp(x-1,y)\times\max\{0,y-f(x)\}$.
其中 $f(x)$ 表示满足 $1\le z<x$ 且 $a_xr<a_z\le a_x$ 的 $????$ 的个数。单组数据时间复杂度:$O(n^2)$

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int M=998244353;
int n,k,r,a[5005],dp[5005][5005];
void solve(){
    memset(dp,0,sizeof(dp));
    cin >> n >> k >> r;
    for (int i=1;i<=n;i++) cin >> a[i];
    int p=0; dp[0][0]=1;
    for (int i=1;i<=n;i++){
        while (p+1<i&&a[p+1]+r<=a[i]) ++p;
        int s=i-p-1;
        for (int j=s+1;j<=i;j++)
            dp[i][j]=(1ll*(j-s)*dp[i-1][j]+dp[i-1][j-1])%M;
    }
    cout << dp[n][k] << endl;
}
int main(){
    int T; cin >> T;
    while (T--) solve();
}
```

# 1008 [Shortest Path in GCD Graph](http://acm.hdu.edu.cn/showproblem.php?pid=7240)

题目大意：求给定 $????$ 个点的完全图，两个点之间距离为它们的gcd，$????$ 次询问两个点之间的最短路以及方案数。

容斥原理求1~n之间与x,y同时互质数的个数：

1.将状态存于数组中，大小为 $2^m$ (质因数个数为$m$)，再遍历每个状态，根据状态（状态中二进制1的个数）来交替加上因数个数

2.dfs遍历所有因数，借助参数对各因数进行分类，轮换加减

std:

```c++
#include<bits/stdc++.h>
#define pb push_back
using namespace std;
typedef long long ll;
const int maxn=1e7+5;
typedef vector<int> vi;
int prime[maxn/10],f[maxn],n,Q;
bool p[maxn];
// get primes
void sieve(int n){
    int cnt=0;
    for (int i=2;i<=n;i++){
        if (!p[i]) prime[++cnt]=i,f[i]=i;
        for (int j=1;j<=cnt&&prime[j]*i<=n;j++){
            p[prime[j]*i]=1;
            f[prime[j]*i]=prime[j];
            if (i%prime[j]==0) break;
        }
    }
}   
int m,ret; 
set<int> S;
vi a;
// get coprime
void pv(int x){
    while (x>1){
        int y=f[x];
        if (S.find(y)==S.end()) S.insert(y);
        while (x%y==0) x/=y;
    }
}
void dfs(int x,int s,int o){
    if (x==m){
        ret+=o*(n/s);
        return;
    }
    dfs(x+1,s,o);
    if (s<=n/a[x]) dfs(x+1,s*a[x],-o);
}
int calc(int x,int y){
    S.clear(); pv(x); pv(y);
    a.clear(); for (auto x:S) a.pb(x);
    m=a.size(); ret=0; 
    dfs(0,1,1);
    return ret;
}
int main(){
    ios::sync_with_stdio(false);
    cin >> n >> Q;
    sieve(n);
    while (Q--){
        int x,y; cin >> x >> y;
        assert(x!=y);
        if (__gcd(x,y)==1){
            cout << 1 << ' ' << 1 << endl;
            continue;
        }
        int ans=calc(x,y);
        if (__gcd(x,y)==2) ans++;
        cout << 2 << ' ' << ans << endl;
    }
    return 0;
}
```

# 1010 [Sum Plus Product](http://acm.hdu.edu.cn/showproblem.php?pid=7242)

签到题：通过前3个数的实验发现表达式轮换对称，模拟即可

```c++
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
typedef long long ll;
const ll mod=998244353;
const ll maxn=100005;
const ll inf=0x3f3f3f3f;
const double pi=acos(-1);
typedef pair<int,int> pii;
vector<ll> b;
void solve(){
	b.clear();
	int n;
	cin>>n;
	ll x,ans=0;
	for(int i=0;i<n;i++){
		cin>>x;
		b.push_back(x);
	}
	for(int i=0;i<n;i++){
		if(i==0)
		ans=((b[i]+b[i+1])+(b[i]*b[i+1]))%mod,i++;
		else
		ans=(b[i]*ans+b[i]+ans)%mod;
	}
	cout<<ans<<endl;
}
signed main(){
	int TT;
	TT=1;
	read(TT);
	while(TT--){
		solve();
	}
	return 0;
}


```