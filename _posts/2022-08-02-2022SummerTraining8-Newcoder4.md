---
title: '2022 Summer ACM training-Newcoder Vol.4'
date: 2022-08-02
permalink: /posts/2022/08/nc4/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# A [Task Computing](https://ac.nowcoder.com/acm/contest/33189/A)

# D [Jobs (Easy Version)](https://ac.nowcoder.com/acm/contest/33189/D)

# H [Wall Builder II](https://ac.nowcoder.com/acm/contest/33189/H)

# K [NIO's Sword](https://ac.nowcoder.com/acm/contest/33189/K)

题目大意：一个数 $x$ 若满足 $x\equiv i\quad\mod n$ 则可以进入下一层 $i+1$，若不满足可以做如下变换：$x=x\times10+y,y\in[0,9]$ 直到满足条件， $x$ 的初始值是0，问到n层至少经过几次变换

# *L [Black Hole](https://ac.nowcoder.com/acm/contest/33189/L) （科普）

题目大意：一个正多面体每次坍缩都会变成连接所有面中心的一个凸多面体，当其不是正多面体时直接消失，问 $k$ 次坍缩后是否存在，并求边长

题目超纲，但可以了解以下结论：

1. 只有五种正多面体，即正四面体，正六面体，正八面体，正十二面体与正二十面体
2. 此题的坍缩过程是先求内切球半径后利用外接球半径反推得到的

不百度大几何题，百度后水题

# N [Particle Arts](https://ac.nowcoder.com/acm/contest/33189/N)

题目大意：对于一个数组每次操作将两数 $a,b$ 修改为 $a |b$ 与 $a\&b$ ，求经过足够多次修改后所有数的方差趋近值

因为修改操作不能改变每个数在特定二进制位上“1”的数量，在经过多次修改后“1”和“0”会分开，如下
$$
\begin{matrix}
1010\\
1011\\
0101\\
0111
\end{matrix}
\quad\rarr\quad
\begin{matrix}
1111\\
1111\\
0001\\
0000
\end{matrix}
$$
模拟即可

TIPS: 由于答案是由分数形式表示，需要对方差算式进行处理
$$
\begin{align}\sigma^2&=\frac1n\sum_{i=1}^n(x_i-\mu)^2\\
&=\frac1n\sum_{i=1}^n\left(x_i^2-2\mu x_i+\mu^2\right)\\
&=\frac1n\sum_{i=1}^nx_i^2-2\mu\frac1n\sum_{i=1}^n x_i+\mu^2\\
&=\frac1n\sum_{i=1}^nx_i^2-\mu^2=\frac{n\sum_{i=1}^nx_i^2-(\sum_{i=1}^n x_i)^2}{n^2}
\end{align}
$$

```c++
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
const int MAXN = 100005;
ll d[34];
ll ans=0;
vector<ll> prc;
ll gcd(ll x,ll y){
	return y==0?x:gcd(y,x%y);
}

int main(){
	ll n;
	cin>>n;
	for(int i=0;i<n;i++){
		int x,cnt=0;
		cin>>x;
		while(x){
			if(x&1)d[cnt]++;
			cnt++;
			x>>=1;
		}
	}
	bool flag=true;
	while(true){
		flag=true;
		int tmp=1,tp=0;
		for(int i=0;i<15;i++){
			if(d[i]){
				d[i]--;
				tp+=tmp;
				flag=false;
			}
			tmp<<=1;
		}
		prc.push_back(tp);
		if(flag)break;
	}
	ll sum=0;
	ll sum2=0;
	for(int i=0;i<prc.size();i++){
		sum2+=prc[i]*prc[i];
		sum+=prc[i];
		assert(("sum ov",sum2>=0&&sum2<9223372036854775807ll)); 
	}
	ll num=sum2*n-sum*sum;
	assert(("num ov",num>=0&&num<9223372036854775807ll));
	ll cd=gcd(n*n,num); 
	cout<<num/cd<<'/'<<n*n/cd<<endl;
	return 0;
}
```