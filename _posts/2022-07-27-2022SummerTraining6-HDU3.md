---
title: '2022 Summer ACM training-HDU Vol.3'
date: 2022-07-27
permalink: /posts/2022/07/2022SummerTraining6-HDU3/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# 1002 [Boss Rush](http://acm.hdu.edu.cn/showproblem.php?pid=7163)

题目大意：有 $n$ 个技能与血量为 $H$ 的BOSS，每个技能只能放一次且需要等待冷却时间 $t_i$ 才可再放技能（ $t+t_i$ 帧），技能效果为在当前帧与接下来共 $len_i$ 帧造成伤害$d_{i,j}$ ，求击败BOSS最短时间

根据范围 $n\le18$ 可以考虑搜索所有技能使用的子集，并利用二分求出最短时间

然后我们考虑如何二分，因为技能对BOSS的伤害是一段区间，所以我们用记忆化搜索得到本次时间内按某一顺序击败BOSS的最短时间
然后因为伤害是连续的，我们可以考虑用前缀和得到该技能施展某一秒时的总伤害，可以优化时间

```c++
#include<cstdio>
typedef long long ll;
const int N=18,M=100005;
int Case,n,i,j,S,t[N],d[N],l,r,ans,mid,sum[(1<<N)+1];
ll hp,f[(1<<N)+1],dmg[N][M];
inline void up(ll&a,ll b){a<b?(a=b):0;}
bool check(int T){
    int S,i;
    for(S=0;S<1<<n;S++)f[S]=-1;
    f[0]=0;
    for(S=0;S<1<<n;S++){  
        ll w=f[S];
        if(w<0)continue;
        if(w>=hp)return 1;
        int cur=sum[S];
        if(cur>T)continue;
        for(i=0;i<n;i++)if(!(S>>i&1)){
            if(cur+d[i]-1<=T)up(f[S|(1<<i)],w+dmg[i][d[i]-1]);
            else up(f[S|(1<<i)],w+dmg[i][T-cur]);
        }
    }
    return 0;
}
int main(){
    scanf("%d",&Case);
    while(Case--){
        scanf("%d%lld",&n,&hp);
        ans=-1, l=r=0;
        for(i=0;i<n;i++){
            scanf("%d%d",&t[i],&d[i]);
            r+=t[i]+d[i]-1;
            for(j=0;j<d[i];j++)scanf("%lld",&dmg[i][j]);
            for(j=1;j<d[i];j++)dmg[i][j]+=dmg[i][j-1];
        }
        for(S=1;S<1<<n;S++)sum[S]=sum[S-(S&-S)]+t[__builtin_ctz(S&-S)];
        
        // 二分答案 
        while(l<=r){ 
            mid=(l+r)>>1;
            if(check(mid))r=(ans=mid)-1;else l=mid+1;
        }
        printf("%d\n",ans);
    }
}

```

# 1003 [Cyber Language](http://acm.hdu.edu.cn/showproblem.php?pid=7164)

签到题：

```c++
#include <iostream>
#include <sstream>
using namespace std;
int main(){
	int t;
	cin>>t;
	cin.get();
	string s;
	while(t--){
		getline(cin,s);
		stringstream ss(s);
		string str;
		while(ss>>str){
			cout<<char(str[0]-('a'-'A'));
		}
		cout<<endl;
	}
	
	return 0;
} 
```

# 1009 [Package Delivery](http://acm.hdu.edu.cn/showproblem.php?pid=7170)

题目大意：

有n个区间，每次最多取k个区间且所有区间包含某一个数，求最小操作次数。

考虑将所有区间分别按照 $rb$（区间上界）和 $lb$（区间上界）从小到大排序 ( $pq_1, pq_2$ )，每次取出 $pq_1$ 的一个元素 $e$ 并取出 $pq_2$ 符合要求的所有区间（均包含 $e$ ），且**将取出的所有区间排序**（按 $rb$ 排序）并依次**打上标记**与剔除（ $k$个一组，由于存在一次取出大于$k$个的情况）

**排序可以仅排序下标，方便进行操作**

标程：

```c++
#include<cstdio>
#include<algorithm>
#include<vector>
#include<queue>
using namespace std;
typedef pair<int,int>P;
const int N=100005;
int Case,n,k,i,j,t,ans,ql[N],qr[N],del[N];
P e[N];
priority_queue<P,vector<P>,greater<P> >q;
inline bool cmpl(int x,int y){return e[x].first<e[y].first;}
inline bool cmpr(int x,int y){return e[x].second<e[y].second;}
int main(){
  scanf("%d",&Case);
  while(Case--){
    scanf("%d%d",&n,&k);
    for(i=1;i<=n;i++){
      scanf("%d%d",&e[i].first,&e[i].second);
      ql[i]=i;
      qr[i]=i;
      del[i]=0;
    }
    sort(ql+1,ql+n+1,cmpl);
    sort(qr+1,qr+n+1,cmpr);
    for(ans=0,i=j=1;i<=n;i++){
      if(del[qr[i]])continue;
      while(j<=n&&e[ql[j]].first<=e[qr[i]].second){
        q.push(P(e[ql[j]].second,ql[j]));
        j++;
      }
      ans++;
      for(t=1;t<=k;t++){
        if(q.empty())break;
        del[q.top().second]=1;
        q.pop();
      }
    }
    printf("%d\n",ans);
  }
}
```

# 1011 [Taxi](http://acm.hdu.edu.cn/showproblem.php?pid=7172)

题目大意：平面坐标系下有 $n$ 个点，有 $k$ 次询问，每次给出任意一点 $(x^\prime,y^\prime)$ ，求
$$
\max_{i=1\dots n}\min\{w_i,|x^\prime-x_i|+|y^\prime-y_i|\}
$$
不考虑 $w_i$ 的影响时，考虑**化简**最大曼哈顿距离$\max d_i$
$$
\begin{align}
\max_{i=1\dots n} d_i&=\max_{i=1\dots n}\{|x^\prime-x_i|+|y^\prime-y_i|\}\\
&=\max_{i=1\dots n}\{x^\prime-x_i+y^\prime-y_i,x_i-x^\prime+y^\prime-y_i,x^\prime-x_i+y_i-y^\prime,x_i-x^\prime+y_i-y^\prime\}
\end{align}
$$
观察可知可存 $x_i+y_i,x_i-y_i$ 最大值与最小值间接求$\max d_i$

将权值 $w_i$ 考虑进来，先将点**按照 $w_i$ 排序**，求出**后缀** $x_i+y_i,x_i-y_i$ **最大值与最小值**，**二分**求解

原理：选取按 $w$ 排序后的第 $k$ 个城镇， $O(1)$ 求出给定点 $(x^\prime, y^\prime)$ 到第 $k\dots n$ 个城镇的距离最大值
$d$，有两种情况：

  		1) $w_k$ < $d$，那么第 $k\dots n$ 个城镇对答案的贡献至少为 $w_k$。用 $w_k$ 更新答案后，由于第 $1\dots k$ 个
  	   城镇的 $w$ 值均不超过 $w_k$ ，因此它们不可能接着更新答案，考虑范围缩小至 $[k + 1, n]$。
  		2) $w_k$ ≥ $d$，那么第 $k\dots n$ 个城镇对答案的贡献为 $d$。用 $d$ 更新答案后，考虑范围缩小至
  	   $[1, k − 1]$。

在 $k$ 取 $mid$ 时，迭代次数缩小到 $O(\log n)$，二分得出答案。

标程：

```c++
#include<cstdio>
#include<algorithm>
using namespace std;
const int N=100005,inf=2100000000;
int Case,n,m,i,x,y,a[N],b[N],c[N],d[N];
struct E{int x,y,w;}e[N];
inline bool cmp(const E&a,const E&b){return a.w<b.w;}
inline void up(int&a,int b){a<b?(a=b):0;}
int main(){
  scanf("%d",&Case);
  while(Case--){
    scanf("%d%d",&n,&m);
    for(i=1;i<=n;i++)scanf("%d%d%d",&e[i].x,&e[i].y,&e[i].w);
    sort(e+1,e+n+1,cmp);
    a[n+1]=b[n+1]=c[n+1]=d[n+1]=-inf;
    for(i=n;i;i--){
      a[i]=max(a[i+1],-e[i].x-e[i].y);
      b[i]=max(b[i+1],-e[i].x+e[i].y);
      c[i]=max(c[i+1],e[i].x-e[i].y);
      d[i]=max(d[i+1],e[i].x+e[i].y);
    }
    while(m--){
      scanf("%d%d",&x,&y);
      int l=1,r=n,mid,tmp,ans=0;
      while(l<=r){
        mid=(l+r)>>1;
        tmp=x+y+a[mid];
        up(tmp,x-y+b[mid]);
        up(tmp,-x+y+c[mid]);
        up(tmp,-x-y+d[mid]);
        if(e[mid].w<tmp){
          l=mid+1;
          up(ans,e[mid].w);
        }else{
          r=mid-1;
          up(ans,tmp);
        }
      }
      printf("%d\n",ans);
    }
  }
}
```

# 1012 [Two Permutations](http://acm.hdu.edu.cn/showproblem.php?pid=7173)

题目大意：将两个 $n$ 排列 $a,b$ 依次push进入 $s$ 的所有可能情况种数（对998244353取模）

做法：

先预处理 $s$ 排除非法情况（s中含2个以上相同元素），考虑DP

由于 $s$ 中仅含2个相同元素，可以按前 $i$ 个元素含有 $j$ 个相同元素转移状态，状态数为 $O(n)$

或者如下用 $j$ 标记 $a,b$ , 每次根据匹配情况（4种）进行转移

```c++
#include<bits/stdc++.h>
using namespace std;

#define IOS ios::sync_with_stdio(0),cin.tie(0)
#define endl '\n'

typedef long long ll;

const ll mod = 998244353;
const int N = 2e6 + 10;

int n;
int a[N], b[N], c[N], cnt[N], pos[N][2];

ll f[N][2];

void solve(){

    cin >> n;
    for (int i = 1;i <= n;++i)cin >> a[i], pos[a[i]][0] = i;
    for (int i = 1;i <= n;++i)cin >> b[i], pos[b[i]][1] = i;

    memset(cnt + 1, 0, n * 4);
    for (int i = 1;i <= n * 2;++i)cin >> c[i], ++cnt[c[i]];
    if (*max_element(cnt + 1, cnt + 1 + n) > 2){
        cout << 0 << endl;
        return;
    }

    for (int i = 1;i <= n * 2;++i)
        f[i][0] = f[i][1] = 0;
    f[1][!(a[1] == c[1])] = 1, f[1][b[1] == c[1]] = 1;

    for (int i = 2, j;i <= n * 2;++i){

        if (pos[c[i - 1]][0] + 1 == pos[c[i]][0])f[i][0] = (f[i][0] + f[i - 1][0]) % mod;
        if (pos[c[i - 1]][1] + 1 == pos[c[i]][1])f[i][1] = (f[i][1] + f[i - 1][1]) % mod;

        j = i - pos[c[i - 1]][0];
        if (b[j] == c[i])f[i][1] = (f[i][1] + f[i - 1][0]) % mod;
        j = i - pos[c[i - 1]][1];
        if (a[j] == c[i])f[i][0] = (f[i][0] + f[i - 1][1]) % mod;
    }
    cout << (f[n * 2][0] + f[n * 2][1]) % mod << endl;
}

int main(){

    IOS;
    int T;
    cin >> T;
    while (T--){
        solve();
    }

    return 0;
}
```