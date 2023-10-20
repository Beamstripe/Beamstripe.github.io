---
title: '2022 Summer ACM training-Newcoder Vol.1'
date: 2022-07-18
permalink: /posts/2022/07/nc1/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# A题：[ Villages: Landlines](https://ac.nowcoder.com/acm/contest/33186/A)

题目大意：以一个区间为起点向外扩散至所有区间，求未被覆盖区间总长

实际上仅需要记录所有区间最大上界 $ub$ 与最小下界 $lb$ 后，将所有区间按上界排序，计算覆盖总长 $len$ 后从 $ub$ 与 $lb$ 中减去即可，即 $ans=rb-lb-len$

```c++
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
using namespace std;
typedef long long ll;
typedef pair<ll,ll> pll;
vector<pll> v;
const ll inf=0x3f3f3f3f3f3f3f3f;
bool cmp2(pll a,pll b){
	if(a.second!=b.second)
	return a.second<b.second;
	else return a.first<b.first;
}
int main(){
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		ll x,y;
		cin>>x>>y;
		v.push_back(pll(x-y,x+y));
	}
	sort(v.begin(),v.end(),cmp2);
	ll rb=v[n-1].second;
	sort(v.begin(),v.end());
	ll lb=v[0].first;
	ll ans=0,now=-inf;
	for(int i=0;i<n;i++){
		if(v[i].first>now){
			ans+=v[i].second-v[i].first;
			now=v[i].second;
		}else if(v[i].second>now){
			ans+=v[i].second-now;
			now=v[i].second;
		}
	}
//	cout<<ans<<endl;
	cout<<rb-lb-ans<<endl;
	return 0;
}
```

# C题：[Grab the Seat!](https://ac.nowcoder.com/acm/contest/33186/C)

题目大意：

对于一个座位（整点），当且仅当该点与屏幕 $(0,1)\rarr(0,m)$ 上所有点的连线上均不包含其他人所在点时是满足要求的。求q次询问变动位置后满足要求座位个数。

![image-20220720130927011](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220720130927011.png)

以 $(0, 1)$ 为例，所有线段都是从某个点往右上方走。
如果按 $y$ 从小到大加入线段，不难发现斜率大的线段一旦加入，就一直
会覆盖斜率小的线段。对于某个 $y$，要找到最小的合法 $x$，那么肯定取的是
前面所有线段中斜率最大的线段。所以对 $y$ 正着扫一遍，在扫的同时维护一
下最大的斜率，即可得出每一列只考虑经过 $(0, 1)$ 的线段的答案。
$(0, m)$ 同理，倒着扫一遍维护斜率最小（绝对值最大）的线段即可。每
个 $y$ 在两种情况取 $\min$ 就是答案。

![c-ex](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/C-ex.png)

标程：

```c++
#include <bits/stdc++.h>
 
const int N = 200010;
using ll = long long;
 
struct frac{
    ll x, y;
 
    explicit frac(ll x = 0, ll y = 1):x(x), y(y){
        if (y < 0){
            this -> x = -this -> x;
            this -> y = -this -> y;
        }
    }
 
    bool operator < (const frac &f)const{
        return x * f.y < y * f.x;
    }
 
    bool operator > (const frac &f)const{
        return x * f.y > y * f.x;
    }
};
 
std::vector <int> vec[N];
int x[N], y[N], min[N];
 
int main(){
    int n, m, k, q;
    scanf("%d%d%d%d", &n, &m, &k, &q);
    for (int i = 0; i < k; ++ i){
        scanf("%d%d", &x[i], &y[i]);
    }
    while (q --){
        int pos;
        scanf("%d", &pos);
        -- pos;
        scanf("%d%d", &x[pos], &y[pos]);
        for (int i = 0; i < N; ++ i){
            vec[i].clear();
        }
        for (int i = 0; i < k; ++ i){
            vec[y[i]].emplace_back(x[i]);
        }
        for (int i = 1; i <= m; ++ i){
            min[i] = n + 1;
        }
        frac max(-1, 0);
        for (int i = 1; i <= m; ++ i){
            if (i == 1){
                for (auto u : vec[i]){
                    min[i] = std::min(min[i], u);
                }
                continue;
            }
            for (auto u : vec[i]){
                max = std::max(max, frac( i - 1, u - 0));
            }
            if (max.y != 0){
                ll x_pos = max.y * (i - 1);
                x_pos = (x_pos + max.x - 1) / max.x;
                if (x_pos < min[i]){
                    min[i] = x_pos;
                }
            }
        }
        max = frac(-1, 0);
        for (int i = m; i >= 1; -- i){
            if (i == m){
                for (auto u : vec[i]){
                    min[i] = std::min(min[i], u);
                }
                continue;
            }
            for (auto u : vec[i]){
                max = std::max(max, frac( m - i, u - 0));
            }
            if (max.y != 0){
                ll x_pos = max.y * (m - i);
                x_pos = (x_pos + max.x - 1) / max.x;
                if (x_pos < min[i]){
                    min[i] = x_pos;
                }
            }
        }
        ll ans = 0;
        for (int i = 1; i <= m; ++ i){
            ans += min[i] - 1;
        }
        printf("%lld\n", ans);
    }
    return 0;
}
```

# D题：[Mocha and Railgun](https://ac.nowcoder.com/acm/contest/33186/D)

题目大意：给定一个圆和严格位于圆内的一点 P
Mocha 会从点 P 向任意角度发射一个长度为 2d 的电磁炮
电磁炮底边的中点为点 P 且两端位于圆内
询问单次发射能摧毁的最大圆弧长

电磁炮如图所示：

![image-20220720135112602](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220720135112602.png)

1. 求得弧长L与弦长l的关系：$L=2r\arcsin(\frac{l}{2r})$
2. 由定理：过圆内定点的最大弦长为圆的直径可知电磁炮方向需要与$OQ$方向垂直
3. 由图求得弦长

![image-20220720141229290](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220720141229290.png)



```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const double PI = acos(-1);
int main()
{
	int t;
	cin>>t;
	while(t--)
	{
		double r,x,y,d;
		cin>>r>>x>>y>>d;
		double oq=sqrt(x*x+y*y);
		double len1=sqrt(r*r-(d+oq)*(d+oq));
		double len2=sqrt(r*r-(d-oq)*(d-oq));
		double len3=sqrt(2*d*2*d+(len2-len1)*(len2-len1));
//		double sita=asin(len3/2/r)*180/3.1415926;
//		double hc=2*PI*r*(double)(2*sita/(double)360.0);
        double ans=2*asin(len3/2/r)*r;
		printf("%.8lf\n",ans);
	}
	return 0;
}

```

# G题：[ Lexicographical Maximum](https://ac.nowcoder.com/acm/contest/33186/G)

题目大意：求区间字典序最大的数

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
const int maxn=209;
signed main(){
	string s;int flag=0;
	cin>>s;
	int n=s.size();
	for(int i=0;i<n-1;i++){
		if(s[i]!='9'){
			flag=1;
		}
	}
	if(n==1){
		cout<<s<<endl;return 0;
	}
	if(flag==0){
		cout<<s;
	}
	else{
		for(int i=0;i<n-1;i++){
			cout<<9;
		}
	} 
	cout<<endl;
	return 0;
}
```

# I题：[ Chiitoitsu](https://ac.nowcoder.com/acm/contest/33186/I)

由于初始手牌中每种牌最多两张，因此最优策略是，若摸到的牌能凑成
对子则**丢弃单牌**，否则**丢弃摸到的牌**。
考虑 DP 求期望，令 $f_{s,r}$ 表示当前手牌中有 $s$ 张单牌且牌堆中剩余 $r$ 张
牌时达成七对子的期望轮数，则有：
$$
f(s,r)=
\begin{cases}1+\frac{r-3}rf_{1,r-1}&(s=1)\\
1+\frac{3s}rf_{s-2,r-1}+\frac{r-3s}rf_{s,r-1}&(s>1)\end{cases}
$$


对于给定的初始手牌，设其单牌数量为 $s_0$，则 $f_{s_0,136−13}$ 即为答案。
时间复杂度 $O(7 × 136 + T)$ 。

标程：

```c++
#include<bits/stdc++.h>
using namespace std;
#define M 200
const int mo=1000000007;
int Inv[M];
void init(){
	int i;
	Inv[1]=1;
	for (i=2;i<M;i++){
		Inv[i]=1LL*(mo-mo/i)*Inv[mo%i]%mo;
	}
}
int cnt[40];
int tr(char x,char y){
	int r=x-'1';
	if (y=='p'){
		r+=9;
	}else if (y=='s'){
		r+=18;
	}else if (y=='z'){
		r+=27;
	}
	return r;
}
long long dp[20][200];
long long dfs(int a,int b){
	if (a<=0 || a*3>b){
		return 0;
	}
	if (dp[a][b]!=-1){
		return dp[a][b];
	}
	return dp[a][b]=((dfs(a,b-1)*(b-a*3)+dfs(a-2,b-1)*a*3)%mo*Inv[b]+1)%mo;
}
char s[30];
int main(){
	init();
	int Case,Tt,i;
	scanf("%d",&Case);
	memset(dp,-1,sizeof(dp));
	for (Tt=1;Tt<=Case;Tt++){
		scanf("%s",s);
		memset(cnt,0,sizeof(cnt));
		for (i=0;i<26;i+=2){
			int x=tr(s[i],s[i+1]);
			cnt[x]++;
		}
		int cnt1=0;
		for (i=0;i<40;i++){
			if (cnt[i]==1){
				cnt1++;
			}
		}
		printf("Case #%d: %lld\n",Tt,dfs(cnt1,4*34-13));
	}
	return 0;
}
```



# J题：[Serval and Essay](https://ac.nowcoder.com/acm/contest/33186/J)

题目大意：将无自环重边的有向图中一个点染色，当且仅当一个点的父节点全部被染色后（所有入边的起点）子节点才可被染色，求在染色点的总数最大值。

设 $u$ 点入边起点集合 $I_u$，$S_u$ 为以 $u$ 为初始节点形成的染色点集合，集合 $T=V$（全集）。每次考虑所有 $T$ 中的点 $v$，若存在 $T$ 中的点 $u$ 使得 $I_v ⊆ S_u$，则令 $S_u\larr S_u\cap S_v$ ，并从 $T$ 中去掉 $v$，重复到不存在满足条件的 $u$, $v$ 为止。

实现：带权并查集维护染色点数量，每次合并（ $x$ 合并到 $y$ 上）将并查集的根节点增加 $x$ 的相应出边（非 $y$ ），并将对应 $x$ 的入边删除，加入过程中入度更新后为1的点到队列中（入度为0的点不能进入队列）

```c++
#include <bits/stdc++.h>
#define maxn 200086
 
using namespace std;
 
int t, n;
set<int> v[maxn], w[maxn];
int fa[maxn], siz[maxn];
 
int find(int x){
	return x == fa[x] ? x : fa[x] = find(fa[x]);
}
 
int main(){
	scanf("%d", &t);
	for(int T = 1;T <= t;T++){
		scanf("%d", &n);
		for(int i = 1;i <= n;i++) v[i].clear(), w[i].clear(), fa[i] = i, siz[i] = 1;
		for(int i = 1;i <= n;i++){
			int k;
			scanf("%d", &k);
			while(k--){
				int x;
				scanf("%d", &x);
				v[x].insert(i);
				w[i].insert(x);
			}
		}
		queue<pair<int, int> > q;
		for(int i = 1;i <= n;i++) if(w[i].size() == 1) q.push({i, *w[i].begin()});
		while(!q.empty()){
			int x = find(q.front().first), y = find(q.front().second);q.pop();
			if(x == y) continue;
			if(v[x].size() > v[y].size()) swap(x, y), swap(w[x], w[y]);
			fa[x] = y, siz[y] += siz[x];
			for(auto i : v[x]){
				int to = find(i);
				if(to == y) continue;
				v[y].insert(to), w[to].erase(x), w[to].insert(y);
				if(w[to].size() == 1) q.push({to, *w[to].begin()});
			}
		}
		int ans = 0;
		for(int i = 1;i <= n;i++) ans = max(ans, siz[find(i)]);
		printf("Case #%d: %d\n", T, ans);
	}
}
/*
1
5
2 4 5
1 3
1 1
1 5
0
*/
```
