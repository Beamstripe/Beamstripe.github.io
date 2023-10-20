---
title: '2022 Summer ACM training-Newcoder Vol.9'
date: 2022-08-15
permalink: /posts/2022/08/nc9/
tags:
  - Chinese post
  - ACM
  - Algorithm
---
![image-20220815185539452](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220815185539452.png)

# A [Car Show](https://ac.nowcoder.com/acm/contest/33194/A)

签到题：给定一个长为n的包含1,2,...,m的序列，求有多少区间[L,R]包含所有1,2,...,m。

双指针（滑动窗口）

```c++
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <functional>
#include <cstring>
#include <string>
#include <cmath>
#include <bitset>
#include <iomanip>
#include <set>
#include <ctime>
#include <cassert>
#include <complex>
#include <cstdio>
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
const ll maxn=100015;
const ll inf=0x3f3f3f3f;
const double pi=acos(-1);
typedef pair<int,int> pii;
int a[maxn];
map<int,int> mp;
void solve(){
	int n,m;
	cin>>n>>m;
	for(int i=1;i<=n;i++){
		cin>>a[i];
	}
	int l=1,r=1,cnt=0;
	ll ans=0;
	mp[a[1]]++;cnt++;
	while(l<=n&&r<=n){
		if(cnt==m){
			ans+=n-r+1;
			mp[a[l]]--;
			if(mp[a[l]]==0){
				--cnt;
			}
			l++;
			continue;
		}
		if(r<n){
			r++;
			if(mp[a[r]]==0){
				++cnt;
			}
			mp[a[r]]++;
		}else break;
	}
	cout<<ans<<endl;
}
signed main(){
	ios::sync_with_stdio(false),cin.tie(0);
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

# B [Two Frogs](https://ac.nowcoder.com/acm/contest/33194/B)

题意：

河道里有 $n$ 个荷叶排成一排，从第$i$ (<n)个荷叶出发可以跳到第 $(i,i+a_i]$ 个荷叶上，有两只青蛙从第 $1$ 个荷叶出发，每一步都独立地等概率随机地跳向后边的荷叶，求两只青蛙以相同步数到达第 $n$ 个荷叶的概率。

此题并不是将跳跃方案数求出，需要逐次转移概率

$f_{i,s}$ 表示从第 $1$个荷叶出发恰好 $s$ 次跳到第 $i$ 个荷叶的概率。

考虑向后转移，$f_{i,s}$ 对$f_{j,s+1}$ $(i<j≤i+a_i)$有$f_{i,s}∕a_i$ 的贡献，前缀和优化即可

```c++
#include <iostream>
using namespace std;
typedef long long ll;
const ll mod=998244353;
const int maxn=8015;
ll dp[maxn][maxn];
ll inv[maxn];
ll a[maxn];
int main(){
	ios::sync_with_stdio(false),cin.tie(0);
	int n;
	cin>>n;
	inv[1]=1;
	for(int i=2;i<=n;i++){
		inv[i]=(mod-mod/i)*inv[mod%i]%mod;
	}
	for(int i=1;i<n;i++){
		cin>>a[i];
	}
	dp[1][0]=1;
	dp[2][0]=mod-1;
	for(int i=1;i<n;i++){
		for(int j=0;j<i;j++){
			dp[i][j]=(dp[i][j]+dp[i-1][j])%mod;
			ll tmp=(dp[i][j]*inv[a[i]])%mod;
			dp[i+1][j+1]=(dp[i+1][j+1]+tmp)%mod;
			dp[i+1+a[i]][j+1]=(dp[i+1+a[i]][j+1]-tmp)%mod;
		}
	}
	ll ans=0;
	for(int i=0;i<n;i++){
		dp[n][i]=(dp[n][i]+dp[n-1][i])%mod;
		ll tmp=dp[n][i];
		ans=(ans+tmp*tmp%mod)%mod;
	}
	cout<<ans<<endl;
	return 0;
} 
```

# G [Magic Spells](https://ac.nowcoder.com/acm/contest/33194/G)

题意：

给定 $k$个字符串 $S_1,S_2,...,S_k$，求有多少个本质不同的公共回文子串。

> 回文自动机（回文树）
>
> **定义0号节点为偶数长度根，1号节点为奇数长度根**。
>
> $fail[u]$ 点 $u$ 后缀边指向的点
>
> $len[u]$ 点 $u$ 所表示的回文串长度
>
> $trie[u][c]$ 点 $u$ 前后各增加一个字符c得到的点
>
> $num[i]$表示以该位置结尾的回文串的个数
>
> $last$ 指向最长回文子串的右端点（必定存在，即一个字符情况）

设置一个 $now$ 数组存储每个回文树的结点，再dfs求出对于每个字符串是否均满足该回文树结点

注意起始结点为0(偶回文串)或1(即回文串)，即now需分别设置为全0与全1，各跑一遍dfs

```c++
#include<bits/stdc++.h>
#define Inf 0x3f3f3f3f
using namespace std;
typedef long long LL;
typedef pair<int,int> P;
const int MAXX=300005;

struct PamNode{
    int ch[26];
    int num;//该位置为结尾的回文串个数
    int sum;//该节点表示的回文串出现的数量
    int fail,len;
    PamNode(){memset(ch,0,sizeof(ch));num=fail=len=sum=0;}
};
char s[MAXX];//the string
struct PAM{
    PamNode pam[MAXX];
    int pam_cnt,last;
    int len;
    LL ans[4];
    PAM(){
        pam[0].len=0;pam[1].len=-1;
        pam[0].fail=1;pam[1].fail=0;
        last=0;pam_cnt=1;
    }
    inline int getfail(int id,int las){
        while(s[id-pam[las].len-1]!=s[id])
            las=pam[las].fail;
        return las;
    }
    void add(int id){
        int p=getfail(id,last);
        if(!pam[p].ch[s[id]-'a']){
            pam[++pam_cnt].len=pam[p].len+2;
            int jj=getfail(id,pam[p].fail);
            pam[pam_cnt].fail=pam[jj].ch[s[id]-'a'];
            pam[pam_cnt].num=pam[pam[pam_cnt].fail].num+1;
            pam[p].ch[s[id]-'a']=pam_cnt;
        }
        last=pam[p].ch[s[id]-'a'];
    }
    void build(){
        len=strlen(s+1);
        for(int i=1;i<=len;++i){
            add(i);
            //do something or not
        }
    }
}pamm[6];

int k,now[6];
LL ans=0LL;

void dfs(int jj[]){
    int kk[6];
    for(int i=1;i<=k;++i)
        kk[i]=jj[i];
    for(int j=0;j<26;++j){
        int flag=0;
        for(int i=1;i<=k;++i){
            if(pamm[i].pam[jj[i]].ch[j]){
                kk[i]=pamm[i].pam[jj[i]].ch[j];
                ++flag;
            }
            else
                break;
        }
        if(flag==k){
            dfs(kk);
            ++ans;
        }
    }
}

inline void getans(){
    for(int i=1;i<=k;++i)
        now[i]=0;
    dfs(now);
    for(int i=1;i<=k;++i)
        now[i]=1;
    dfs(now);
}

inline void solve(){
	scanf("%d",&k);
    for(int i=1;i<=k;++i){
        getchar();
        scanf("%s",s+1);
        pamm[i].build();
    }
	
    getans();
	printf("%lld\n",ans);
}

signed main(){
//	LL t;scanf("%lld",&t);
//	while(t--)
		solve();
	
	return 0;
}
```