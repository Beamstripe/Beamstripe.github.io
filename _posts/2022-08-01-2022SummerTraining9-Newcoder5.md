---
title: '2022 Summer ACM training-Newcoder Vol.5'
date: 2022-08-01
permalink: /posts/2022/08/nc5/
tags:
  - Chinese post
  - ACM
  - Algorithm
---


# B [Watches](https://ac.nowcoder.com/acm/contest/33190/B)

简单二分：注意向较大数取值时mid=(l+r+1)/2

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int MAXN = 100005;
int a[MAXN];
int b[MAXN];
int n,m;
bool check(int k){
	for(int i=1;i<=n;i++){
		b[i]=a[i]+k*i;
	}
	sort(b+1,b+n+1);
	int ans=0;
	for(int i=1;i<=k;i++){
		if(b[i]+ans<=m)
		ans+=b[i];
		else{
			return false;
		}
	}
	return true;
}
int main(){
	ios::sync_with_stdio(false),cin.tie(0);
	cin>>n>>m;
	for(int i=1;i<=n;i++){
		cin>>a[i];
	}
	int l=0,r=n;
	while(l<r){
		int mid=(l+r+1)>>1;
		if(check(mid)){
			l=mid;
		}else{
			r=mid-1;
		}
	}
	cout<<l<<endl;
	return 0;
}
```

# C [Bit Transmission](https://ac.nowcoder.com/acm/contest/33190/C)

题目数据有误\题意不明，此题可忽略

赛时WA代码（不改了，反正特判也能过）

```c++
#include <iostream>
using namespace std;
const int MAXN =1e5+15;
int cnt[MAXN];
int f[MAXN][2];
int main(){
    int n;
    ios::sync_with_stdio(false),cin.tie(0);
    while(cin>>n){
        string s;
        int x;
        int cntcs=0;
        for(int i=0;i<n;i++){
            cnt[i]=0;
            f[i][0]=f[i][1]=0;
        }
        int flag=-1;
        for(int i=0;i<3*n;i++){
            cin>>x>>s;
            cnt[x]++;
            if(s[0]=='Y'){
                if(f[x][0])flag=x;
                f[x][1]++;
            }else{
                if(f[x][1])flag=x;
                f[x][0]++;
            }
        }
        for(int i=0;i<n;i++){
            if(cnt[i]>=3)cntcs++;
            else if(cnt[i]>0&&flag!=-1&&cnt[flag]>=3)cntcs++;
        }
        if(cntcs!=n)cout<<-1<<endl;
        else{
            for(int i=n-1;i>=0;i--){
                cout<<(f[x][1]>f[x][0]?1:0);
            }
            cout<<endl;
        }
    }
    return 0;
}
```

# F [A Stack of CDs](https://ac.nowcoder.com/acm/contest/33190/F)

原题：https://blog.csdn.net/weixin_30347009/article/details/95804542

题目大意：求所有圆覆盖后的总周长

枚举每一个圆，看它有多少没有被覆盖。

每一个圆的极角可以拉直成一个长为 $2\pi r$ 的线段

然后套用数学公式，算出一个圆覆盖的范围，求出所有线段未被覆盖的长度

要注意讨论极角小于0的情况

参考代码：

```c++
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
using namespace std;
struct ZYYS {
	double l,r;
} a[5001];
double pi=acos(-1.0);
int n;
double x[1001],y[1001],r[1001],ans;
bool cmp(ZYYS a,ZYYS b) {
	return a.l<b.l;
}
double dist(int i,int j) {
	return sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]));
}
bool contain(int i,int j) {
	if (r[i]-r[j]>=dist(i,j)) return 1;
	return 0;
}
double cal(int o) {
	int i;
	int cnt=0;
	for (i=o+1; i<=n; i++)
		if (contain(i,o)) return 0;
	for (i=o+1; i<=n; i++) {
		if (contain(o,i)||dist(o,i)>=r[i]+r[o]) continue;
		double d=dist(o,i);
		double xt=acos((-r[i]*r[i]+r[o]*r[o]+d*d)/(2.0*d*r[o]));
		double aef=atan2(y[i]-y[o],x[i]-x[o]);
		a[++cnt]=(ZYYS) {
			aef-xt,aef+xt
		};
		if (a[cnt].l<0) a[cnt].l+=2*pi;
		if (a[cnt].r<0) a[cnt].r+=2*pi;
		if (a[cnt].l>a[cnt].r) {
			double p=a[cnt].l;
			a[cnt].l=0;
			a[++cnt].l=p;
			a[cnt].r=2*pi;
		}
	}
	sort(a+1,a+cnt+1,cmp);
	double res=0,now=0;
	for (i=1; i<=cnt; i++) {
		if (a[i].l>now) res+=a[i].l-now,now=a[i].r;
		now=max(now,a[i].r);
	}
	res+=2*pi-now;
	return res*r[o];
}
int main() {
	int i;
	cin>>n;
	for (i=1; i<=n; i++) {
		scanf("%lf%lf%lf",&r[i],&x[i],&y[i]);
	}
	for (i=1; i<=n-1; i++)
		ans+=cal(i);
	ans+=2*pi*r[n];
	printf("%.3lf\n",ans);
}
```



# G [KFC Crazy Thursday](https://ac.nowcoder.com/acm/contest/33190/G)

回文自动机模板：

```c++
#include<iostream>
#include<cstdio>
#include<map>
#include<cstring>
using namespace std;
typedef long long ll;
string s,str;
int len[2000001],n,num[2000001],fail[2000001],last,cur,pos,trie[2000001][26],tot=1;
int getfail(int x,int i)
{
	while(i-len[x]-1<0||s[i-len[x]-1]!=s[i])x=fail[x];
	return x;
}
map<char,ll> mp;
int main()
{
	
	cin>>n>>s;
    fail[0]=1;len[1]=-1;
    for(int i=0;i<=n-1;i++){
    	pos=getfail(cur,i);
        if(!trie[pos][s[i]-'a']){
        	fail[++tot]=trie[getfail(fail[pos],i)][s[i]-'a'];
        	trie[pos][s[i]-'a']=tot;
        	len[tot]=len[pos]+2;
            num[tot]=num[fail[tot]]+1;
		}
        cur=trie[pos][s[i]-'a'];
        last=num[cur];
		mp[s[i]]+=last;	   
	}
	cout<<mp['k']<<' '<<mp['f']<<' '<<mp['c']<<endl;
	return 0;
}
```

# H [Cutting Papers](https://ac.nowcoder.com/acm/contest/33190/H)

无语，题意不明/样例错误，水题

![image-20220801195443076](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220801195443076.png)

# K [Headphones](https://ac.nowcoder.com/acm/contest/33190/K)

简单的抽屉原理

```c++
#include <queue>
#include <set>
#include <iostream>
using namespace std;
int main(){
	int n,m;
	cin>>n>>m;
	if(n-m<=m){
		cout<<-1<<endl;return 0;
	}
	int num=0;
	num=n+1;
	cout<<num<<endl;
	return 0;
} 
```