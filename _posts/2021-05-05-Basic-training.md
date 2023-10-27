---
title: '2021ACMtraining-basics'
date: 2021-05-05
permalink: /posts/2021/05/Basic-training/
tags:
  - preliminary knowledge
  - Chinese post
---

# 2021ACM训练#1：输入输出与适应性训练



## 1.零碎事项：

### 训练安排

### 赛制：ACM

捆绑测试，每次提交均可以看到反馈，有罚时，     以最后一次提交为准，可以看到实时排名。相同通过题数按照答题时间+罚时排名。采用该赛制的比赛有ICPC, CCPC, CodeForces, leetcode周赛与全国编程大赛，牛客小白杯练习赛及挑战赛，传智杯，校赛。

### OI

提交没有反馈，按测试点给分，多次提交没有惩罚，以最后一次提交为准，赛后按照得分排名。采用该赛制的比赛有NOIP，OI, CCF CSP,考研机试，蓝桥杯，牛客OI赛，全国高校计算机能力挑战赛。

### IOI

有反馈，可以看到得分，按测试点给分，多次提交没有惩罚，以最后一次提交为准，可以看到实时排名，按照总得分排名。采用该赛制的比赛有PAT，天梯赛，CCF CCSP，洛谷月赛                    

### 赛事介绍：ICPC, CCPC, 蓝桥杯，天梯赛

## 2.代码入门：

### a).头文件：

#### 万能头文件：<bits/stdc++.h>

#### 常用头文件：

```c++
#include <iostream>
#include <algotithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <stack>
#include <sstream> 
```

### b).空间问题：

函数内变量数组的内存在栈中申请，一般大小为1~2M，支持开的数组比较小，如过大可能爆栈

全局变量、全局数组、静态数组（static)在静态区申请，大小为2G：数组1e6~1e7

malloc和new动态地在堆中申请空间大小（不连续）理论上是硬盘大小

### c).时间问题：

O(1)<O(logn)<O(n)<O(nlogn)<O(n^2^)<O(n^3^)<<O(2^n^)<<O(n!)

1000ms测评机大概进行5e8次计算

| 时间复杂度 | 数据范围 |
| ---------- | -------- |
| O(n)       | 1e8      |
| O(nlogn)   | 1e6      |
| O(n^3/2^)  | 1e5      |
| O(n^2^)    | 5000     |
| O(n^3^)    | 300      |
| O(2^n^)    | 25       |
| O(n!)      | 11       |

### d).常用技巧

memset ---> 0,-1,0x3f

INF ---> 0x3f3f3f3f

非void 记得return

##### 输入输出：

1. 吃回车：getchar() , scanf(), gets() （此时可用cin.ignore()，清空键盘缓存区）; *gets慎用*

2. c格式占位符：%d %lld(%l64d) %lf %u %f %s %c %p(*pt) %e(sci) %x(hex) %o(oct)

3. 保留位数：%.2lf 、setprecision()

4. 多样例：

   ```c++
   //#1: 输入t组样例
   int t;cin>>t;
   while(t--){
   	//......
   }
   //#2: 输入未知多个样例
   int a,b;
   while(cin>>a>>b){}//c++ style
   while(scanf("%d%d")!=EOF){}
   while(~scanf("%d%d")){}//c style
   //#3: 满足条件后终止
   int a;
   while(cin>>a&&a!=0){}
   while(~scanf("%d",&a),a){}
   while(cin>>a){
   	if(!a)break;
   }
   ```

5. string 与 stringstream:

   string 类常用:

   ```c++
   int a=123;
   string s=to_string(a);
   cout<<(s.size()==s.length)<<endl;
   s+="456";
   s=s.substr(1,4); //2345
   a=stoi(s); //a=2345
   ```

   stringstream 类常用:

   ```c++
   #include <sstream>
   string s;
   int n, sum = 0;
   cin>>s;
   int len = s.length();
   for(int i = 0; i < len; i++){
   	s[i] == 'a' ? s[i] = ' ': s[i]; /*stringstream 按空格切割流*/
   }
   stringstream s s(s);
   while(ss >> n) sum += n;
   cout<< sum <<endl;
   return 0;
   ```

6. 输入输出加速

   输入速度：cin << scanf() < 关流cout < read() < fread()

   关流操作：ios::sync_with_stdio(false),cin.tie(0);

   注意：关流后不可使用c的输入输出即scanf,printf,puts,gets

   换行加速：#define endl '\n'    ---     gets():<将最后一个'\n'改为'\0'>   ---puts(): <将最后一个'\0'改为'\n'> 

   快读：

   ```c++
   template <typename T> inline void read(T &x){
   	static char ch=getchar();
   	x=0;
   	T sgn=1;
   	while(ch>'9'||ch<'0'){
   		if(ch=='-')sgn=-1;ch=getchar();
   	}
   	while(ch>='0'&&ch<='9')x=x*10+ch-'0',ch=getchar();
   	x=sgn*x;
   } //int long long char
   template <typename T> inline void out(T a)  
   {  
   	if(a>=10)out(a/10);  
       putchar(a%10+'0');  
   }
   ```

7. 离线与在线

   离线：先暂时储存输入数据，待输入结束后处理

   在线：动态读取并处理数据，立即输出（适合多组样例）

##### 结构体：

```c++
struct node {
	double x,y;
	node() { x = 1, y = 1; }//空参构造
	node(double xx,double yy) : x(xx), y(yy) {}//满参构造
	bool operator<(const node &A) const {
        if(x != A.x) return x < A.x;
        return y < A.y;
	}//#1：重载运算符，在sort时会自动调用
}
bool cmp(node a,node b){
    if(a.x != b.x) return a.x < b.x;
        return a.y < b.y;//#2：写cmp，在sort时传入该函数指针
}
```

二分法：

使用对象：顺序表（先sort)

（返回指针类型） lower_bound(第一个元素指针，最后一个元素下一位置指针，要查找的元素)

lower_bound 找第一个 >= element 的位置; upper_bound > ;

STL中判断char类型的函数

```c++
bool
isalnum(); //字母数字
isalpha();//字母
iscntrl();//控制符
isdigit();//数字
islower();//小写字母
isprint();//打印字符（不包括换行符）
isspace();//空格
isupper();//大写字母
isxdigit();//十六进制字符（大小写abcdef与数字）
char tolower(const char &ch);//字母数字
char toupper(const char &ch);//字母数字
```

             xxxxxxxxxx #pragma GCC optimize(3)#include<bits/stdc++.h>#define MAXN 1000005#define MAXM 10000005#define INF 1000000000#define MOD 1000000007#define F first#define S secondusing namespace std;typedef long long ll;typedef pair<int,int> P;int n,m,k,a[MAXN],b[MAXN];P save[2*MAXM];int pa[MAXM],pb[MAXM];int main(){    scanf("%d%d",&n,&m);    for(int i=1;i<=n;i++) scanf("%d",&a[i]);    for(int i=1;i<=m;i++) scanf("%d",&b[i]);    P p=P(0,0),q=P(0,0);    vector<int> va,vb;    memset(pa,0,sizeof(pa));    memset(pb,0,sizeof(pb));    for(int i=1;i<=n;i++)        if(pa[a[i]]) p=P(pa[a[i]],i); else {pa[a[i]]=i; va.push_back(i);}    for(int i=1;i<=m;i++)        if(pb[b[i]]) q=P(pb[b[i]],i); else {pb[b[i]]=i; vb.push_back(i);}    if(p.F!=0&&q.F!=0)    {        printf("%d %d %d %d\n",p.F,p.S,q.F,q.S);        return 0;    }    for(int i=1;i<=20000000;i++) save[i]=P(0,0);    for(int i=0;i<(int)va.size();i++)        for(int j=0;j<(int)vb.size();j++)        {            int sum=a[va[i]]+b[vb[j]];            if(save[sum].F)            {                printf("%d %d %d %d\n",save[sum].F,va[i],min(vb[j],save[sum].S),max(vb[j],save[sum].S));                return 0;            }            save[sum]=P(va[i],vb[j]);        }    puts("-1");    return 0;}c++

