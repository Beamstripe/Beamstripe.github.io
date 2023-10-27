---
title: '2022 Summer ACM training-HDU Vol.2'
date: 2022-07-28
permalink: /posts/2022/07/hd2/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# 1001 [Static Query on Tree](http://acm.hdu.edu.cn/showproblem.php?pid=7150)

题目大意：

一棵树，若从集合A,B中的点分别出发，可以相遇的城市中有多少能到达集合C中的城市。

将A,B中的点到根打上相应标记，对C的子树打上标记。然后统计同时有三个标记的城市数。

```c++
#include <bits/stdc++.h>
using namespace std;

const int N = 200010;
int n, q;
vector<int> e[N];
vector<int> a, b, c;
int l[N], r[N];
int cnt;

namespace lca {
    int dep[N], son[N], sz[N], top[N], fa[N];
    void dfs1(int x) {
        sz[x] = 1;
        son[x] = -1;
        for (auto p : e[x]) {
            if (p == fa[x]) continue;
            fa[p] = x; dep[p] = dep[x] + 1;
            dfs1(p);
            sz[x] += sz[p];
            if (son[x] == -1 || sz[son[x]] < sz[p])
                son[x] = p;
        }
    }
    void dfs2(int x, int tv) {
        top[x] = tv;
        if (son[x] == -1) return;
        dfs2(son[x], tv);
        for (auto p : e[x]) {
            if (p == fa[x] || p == son[x]) continue;
            dfs2(p, p);
        }
    }
    void init(int s) {
        fa[s] = -1; dep[s] = 0;
        dfs1(s);
        dfs2(s, s);
    }
    int lca(int x, int y) {
        while (top[x] != top[y])
            if (dep[top[x]] >= dep[top[y]]) x = fa[top[x]];
            else y = fa[top[y]];
        return dep[x] < dep[y] ? x : y;
    }
}

void dfs(int x) {
    l[x] = ++cnt;
    for (int p : e[x]) {
        dfs(p);
    }
    r[x] = cnt;
}

int calc(vector<int> &a, const vector<int> &c) {
    sort(a.begin(), a.end(), [](int x, int y) {
        return l[x] < l[y];
    });
    int left = 0, res = 0;
    for (int cc : c) {
        while (left < (int)a.size() && l[a[left]] < l[cc]) {
            left++;
        }
        int right = left;
        while (right < (int)a.size() && l[a[right]] <= r[cc]) {
            right++;
        }
        for (int i = left; i < right; i++) {
            res += lca::dep[a[i]] - lca::dep[cc] + 1;
        }
        for (int i = left; i < right - 1; i++) {
            res -= lca::dep[lca::lca(a[i], a[i + 1])] - lca::dep[cc] + 1;
        }

        left = right;
    }

    return res;
}

void solve() {
    scanf("%d%d", &n, &q);

    for (int i = 1; i <= n; i++) {
        e[i].clear();
    }
    cnt = 0;

    for (int i = 2; i <= n; i++) {
        int fa;
        scanf("%d", &fa);
        e[fa].push_back(i);
    }
    dfs(1);
    lca::init(1);

    while (q--) {
        {
            int A, B, C;
            scanf("%d%d%d", &A, &B, &C);
            a.assign(A, 0);
            b.assign(B, 0);
            c.assign(C, 0);
        }
        for (int i = 0; i < (int)a.size(); i++) {
            scanf("%d", &a[i]);
        }
        for (int i = 0; i < (int)b.size(); i++) {
            scanf("%d", &b[i]);
        }
        for (int i = 0; i < (int)c.size(); i++) {
            scanf("%d", &c[i]);
        }

        sort(c.begin(), c.end(), [](int x, int y) {
            return l[x] < l[y];
        });
        int pre = 0;
        for (int i = 1; i < (int)c.size(); i++) {
            if (l[c[pre]] <= l[c[i]] && l[c[i]] <= r[c[pre]]) {}
            else {
                pre++;
                c[pre] = c[i];
            }
        }
        c.erase(c.begin() + pre + 1, c.end());

        int ans = calc(a, c) + calc(b, c);
        for (int i : b) {
            a.push_back(i);
        }
        ans -= calc(a, c);
        printf("%d\n", ans);
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        solve();
    }
    return 0;
}
```

# 1002 [C++ to Python](http://acm.hdu.edu.cn/showproblem.php?pid=7151)

签到题：

```c++
#include <iostream>
using namespace std;
int main(){
	string tp="std::make_tuple";
	int t;
	cin>>t;
	while(t--){
		string str;
		cin>>str;
		bool flag;
		for(int i=0;i<str.length();i++){
			flag=true;
			for(int j=0;j<tp.length();j++){
				if(tp[j]==str[i]){
					i++;
				}else{
					flag=false;
					break;
				}
			}
			if(flag){
				i--;continue;
			}
			else putchar(str[i]);
		}
		cout<<'\n';
	}
	return 0;
}
```

# 1003 [Copy](http://acm.hdu.edu.cn/showproblem.php?pid=7152)

题目大意：

对于数组$a$，每次将区间 $[l_i,r_i]$ 的值复制一份插入到 $[r_i+1,r_i+l_i]$ ，在线询问数组在位置 $i$ 的值。

倒序处理所有询问，每次修改使接下来大于 $r_i$的所有答案减去 $r_i-l_i+1$

但是这么处理还是 的。考虑到我们只要答案的异或和，就有，两个相同的查询可以抵消，因此同一位置至多只会查询一次。这样每个位置用 1 bit 的信息表示即可，也就是 bitset。
我们令 dp 第 i 位为 1 表示答案需要对 a[i] 异或。倒着遍历所有操作，如果是查询操作， dp[x] ^= 1 ，如果是修改操作，那么就让 r+1..n 这些比特右移 r-l+1

# 1004 [Keychains](http://acm.hdu.edu.cn/showproblem.php?pid=7153)

询问两个空间圆是否相扣

显然关键问题是如何求圆 A 和 圆 B 所在平面 的两个交点。得到这两个交点后，只要判断是否一个点在圆 B 内部，一个点在圆 B 外部（到圆心的距离小于半径 or 大于半径），就是答案了。
直接算圆和平面交点似乎不太好做，考虑到这两个交点肯定在两个圆所在平面上，因此先求这两个平面的交线，然后求交线和球 A 的交点（假设球 A 的圆心和半径就是圆 A 的半径）。
直线和球的交点的求法，和平面几何的直线和圆交点的求法类似，不细讲了。剩下的都是板子的事情了。

# 1006 [Bowcraft](http://acm.hdu.edu.cn/showproblem.php?pid=7155)

题目大意：

一个人需要借助魔法书升到K级，其中每一本魔法书的升级成功的概率是$\alpha$，失败并回到零级的概率是 $(1-\alpha)*\beta$ 其中 $\alpha = a/A$, $\beta=b/B$ ,a 和 b 是均匀分布生成的随机整数，范围是 $[0,A-1]$ 和 $[0,B-1]$ ，已知使用魔法书时采用最优策略，求升到 K 级需要使用魔法书的期望

令 dp[i] 表示从 0 级升级到 i 级期望的数量。
假设现在等级为 i，买了一本书 (a,b)。
若使用这本书，升到 i+1 级的期望是 $dp[i]+1+(1-\alpha)(1-\beta)\cdot(dp[i+1]-dp[i])+(1-\alpha)\beta \cdot dp[i+1]$
若不使用这本书，升到 i+1 级的期望是 $dp[i+1]+1$
得到 dp 方程：
$$
dp[i+1]=\frac1{AB}\sum_{a,b}min\{dp[i+1]+1,dp[i]+1+(1-\alpha)(1-\beta)\cdot(dp[i+1]-dp[i])+(1-\alpha)\beta \cdot dp[i+1]\}
$$
对于当前等级 $i$ 和一本书 $(a,b)$ ，若要使用此书升级到 $i+1$ 级，不使用的期望 $\ge$ 使用的期望

即$dp[i+1]+1\ge dp[i]+1+(1-\alpha)(1-\beta)\cdot(dp[i+1]-dp[i])+(1-\alpha)\beta \cdot dp[i+1]$

得 $dp[i+1]\ge dp[i]\cdot \frac{\alpha+\beta-\alpha \beta}{\alpha}$

需要使 $\frac{\beta(1-\alpha)}\alpha$ 尽可能小

将所有可能的书（总数为 $AB$，且每种情况的可能性相同）按照 $\frac{\beta(1-\alpha)}\alpha$ 的大小排序，只使用前 $t$ 小的书，列出等式
$$
AB\cdot dp[i+1] = (AB-t)\cdot (dp[i+1]+1)+\sum_{a,b(前t小)}dp[i]+1+(1-\alpha)(1-\beta)\cdot(dp[i+1]-dp[i])+(1-\alpha)\beta \cdot dp[i+1]
$$
化简得
$$
dp[i+1]=\frac{AB+dp[i]\cdot\sum_{a,b(前t小)}\alpha+\beta-\alpha\beta}{t-\sum_{a,b(前t小)}1-a}
$$
枚举t得到 $dp[i+1]$ 的最小值，递推转移得到 $dp[K]$ ，时间复杂度 $O(KAB)$

# 1007 [Snatch Groceries](http://acm.hdu.edu.cn/showproblem.php?pid=7156)

伪阅读理解，建议不读题目看条件

题目大意：n个区间若不发生重叠则处理请求，否则**停止接收请求**（不读题的代价是-8三小时过）

签到，直接将区间按下界排序，判断下个区间上界即可

```c++
#include <iostream>
#include <vector>
#include <algorithm>
//#define int long long
using namespace std;
typedef pair<int,int> pii;
vector<pii> v;
signed main(){
//	ios::sync_with_stdio(false),cin.tie(0);
	int t;
	cin>>t;
	while(t--){
		int n;
		cin>>n;
		v.clear();
		for(int i=0;i<n;i++){
			int x,y;
			cin>>x>>y;
			v.push_back(pii(x,y));
		}
		sort(v.begin(),v.end());
		int cnt=0;
		for(int i=0;i<v.size();i++){
			int r=v[i].second,j=i+1;
			int k=1;
			while(j<v.size()&&v[j].first<=r){
				r=max(r,v[j].second);
				j++;
				k++;
			}
			i=j-1;
			if(k==1){
				cnt++;
			}
			else break;
		}
		cout<<cnt<<'\n';
	}
	return 0;
}
```

# 1009 [ShuanQ](http://acm.hdu.edu.cn/showproblem.php?pid=7158)

题目大意：$P\times Q\equiv 1 \mod M$ 如果M存在则求 $encrypt\times Q\mod M$

由于 $M$ 是 $P\times Q-1$ 的一个大于$P$, $Q$, $encrypt$ 的质因子，可直接线性筛去掉所有小于等于$P$, $Q$, $encrypt$ 的所有因子，若不为1则 $M$ 存在

```c++
#include <iostream>
using namespace std;
typedef long long ll;
const int MAXN=2e6+15;
int cnt=0;
int st[MAXN];
int prime[MAXN]; 
void get_prime(int n) {
    for (int i = 2; i <= n; ++i) {
        if (!st[i]) prime[cnt++] = i; 
        for (int j = 0; prime[j] <= n/i; ++j) {
            st[prime[j] * i] = true;  
            if (i % prime[j] == 0)  break; 
        }                
    }
}

int main(){
	get_prime(2000004);
	ios::sync_with_stdio(false),cin.tie(0);
	int t;
	cin>>t;
	while(t--){
		ll p,q,en;
		cin>>p>>q>>en;
		ll tp=p*q-1;
		for(int i=0;i<cnt;i++){
			if(prime[i]>max(max(p,q),en))break;
			while(tp%prime[i]==0)tp/=prime[i];
		}
		if(tp==1ll)cout<<"shuanQ\n";
		else
		cout<<en*q%tp<<endl;
	}
	return 0;
}

```

# 1011 [DOS Card](http://acm.hdu.edu.cn/showproblem.php?pid=7160)

题目大意：

数组 $a_i$ 中两两匹配的值为$(a_i-a_j)(a_i+a_j)$ 每次询问 $[L_i,R_i]$ 中两两配对后值之和的最大值

一对匹配的值 左 左 右 右
线段树上维护以下8个变量：
区间最大值
区间次大值
区间最小值
区间次小值
选了一对的最大值
选了两对的最大值
（一对的值 剩下的最大值）的最大值
（一对的值 剩下的最小值）的最大值

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
template<class T>inline void MAX(T &x,T y){if(y>x)x=y;}
template<class T>inline void MIN(T &x,T y){if(y<x)x=y;}
template<class T>inline void rd(T &x){
	x=0;char o,f=1;
	while(o=getchar(),o<48)if(o==45)f=-f;
	do x=(x<<3)+(x<<1)+(o^48);
	while(o=getchar(),o>47);
	x*=f;
}
template<class T>inline void print(T x,bool op=1){
	static int top,stk[105];
	if(x<0)x=-x,putchar('-');
	if(x==0)putchar('0');
	while(x)stk[++top]=x%10,x/=10;
	while(top)putchar(stk[top--]+'0');
	putchar(op?'\n':' ');
}
const int M=1e5+5;
int cas,n,m,A[M];
struct node{
	int len;
	ll max1,max2,min1,min2,res1,res2,res1_max,res1_min;
	node(int len=0){
		this->len=len;
		max1=max2=-1e18;
		min1=min2=1e18;
		res1=res2=res1_max=res1_min=-1e18;
	}
	void update_max(ll v){
		if(v>max1)max2=max1,max1=v;
		else if(v>max2)max2=v;
	}
	void update_min(ll v){
		if(v<min1)min2=min1,min1=v;
		else if(v<min2)min2=v;
	}
	node operator +(const node &A)const{
		if(len==0)return A;
		if(A.len==0)return *this;
		
		node T(len+A.len);
		
		T.update_max(max1);
		T.update_max(max2);
		T.update_max(A.max1);
		T.update_max(A.max2);
		
		T.update_min(min1);
		T.update_min(min2);
		T.update_min(A.min1);
		T.update_min(A.min2);
		
		MAX(T.res2,res2);
		MAX(T.res2,A.res2);
		MAX(T.res2,res1+A.res1);
		MAX(T.res2,max1+max2-A.min1-A.min2);
		MAX(T.res2,res1_max-A.min1);
		MAX(T.res2,max1+A.res1_min);
		
		MAX(T.res1,res1);
		MAX(T.res1,A.res1);
		MAX(T.res1,max1-A.min1);
		
		MAX(T.res1_max,res1_max);
		MAX(T.res1_max,A.res1_max);
		MAX(T.res1_max,res1+A.max1);
		MAX(T.res1_max,A.res1+max1);
		if(A.len>=2)MAX(T.res1_max,max1-A.min1+A.max1);
		if(len>=2)MAX(T.res1_max,max1-A.min1+max2);
		
		MAX(T.res1_min,res1_min);
		MAX(T.res1_min,A.res1_min);
		MAX(T.res1_min,res1-A.min1);
		MAX(T.res1_min,A.res1-min1);
		if(A.len>=2)MAX(T.res1_min,max1-A.min1-A.min2);
		if(len>=2)MAX(T.res1_min,max1-A.min1-min1);
		
		return T;
	}
}tree[M<<2];
void build(int l=1,int r=n,int p=1){
	if(l==r){
		tree[p]=node(1);
		tree[p].max1=tree[p].min1=1ll*A[l]*A[l];
		return;
	}
	int mid=l+r>>1;
	build(l,mid,p<<1);
	build(mid+1,r,p<<1|1);
	tree[p]=tree[p<<1]+tree[p<<1|1];
}
node query(int a,int b,int l=1,int r=n,int p=1){
	if(l>b||r<a)return node(0);
	if(l>=a&&r<=b)return tree[p];
	int mid=l+r>>1;
	return query(a,b,l,mid,p<<1)+query(a,b,mid+1,r,p<<1|1);
}

signed main(){
#ifndef ONLINE_JUDGE
//	freopen("jiedai.in","r",stdin);
//	freopen("jiedai.out","w",stdout);
#endif
	rd(cas);
	while(cas--){
		rd(n),rd(m);
		for(int i=1;i<=n;i++)rd(A[i]);
		build();
		while(m--){
			int l,r;
			rd(l),rd(r);
			print(query(l,r).res2);
		}
	}
	return (0-0);
}
```

# 1012 [Luxury cruise ship](http://acm.hdu.edu.cn/showproblem.php?pid=7161)

题目大意：
$$
ans=\min(x+y+z)\\
\begin{cases}
x\in N^+,y\in N^+,z\in N^+\\
365x+31y+7z=C
\end{cases}
$$
7，31，365的最小公倍数为79205，所以 $n$ 中大于79205的部分使用体积为365的物品填充是最优的。剩下的79205以内的部分直接用一维dp解决。

```c++
#include<bits/stdc++.h>
#define ll long long
#define cer(x) cerr<<(#x)<<" = "<<(x)<<'\n'
#define endl '\n'
using namespace std;
ll n;
ll f[79210];

int main(){ 
	ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
	for(int i=0;i<=79205;i++){
		f[i]=1e18+10;
	}
	f[0]=0;
	for(int i=365;i<79205;i+=365){
		f[i]=f[i-365]+1;
	}
	for(int i=31;i<79205;i++){
		f[i]=min(f[i],f[i-31]+1);
	}
	for(int i=7;i<79205;i++){
		f[i]=min(f[i],f[i-7]+1);
	}
	int t; cin>>t;
	while(t--){
		cin>>n;
		if(n%79205==0){ // 每一份
			cout<<n/365<<endl;
			continue;	
		}
		// n%79205!=0
		ll ans=0;
		ans+=n/79205*217; // 每一份用217个365面值的硬币 
		n%=79205; // 缩小n的范围， 
		if(f[n]>1e17){
			cout<<-1<<endl;
		}
		else{
			cout<<f[n]+ans<<endl;
		}	
	}
	return 0;
}

```