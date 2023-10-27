````c++
# 树上数据结构——树链剖分

树链剖分的实质是通过轻重边剖分将树分割为多条链，利用线段树等数据结构来维护每一条链，从而实现树的高效修改与查询。此方法常用于树的批量处理，特别是树形DP经常出现。



## 前置知识

线段树，dfs序，$LCA$



## 实例引入

已知一棵包含 $N$ 个结点的树（连通且无环），每个节点上包含一个数值，需要处理以下共 $M$ 个操作：

- `1 x y z`，表示将树从 $x$ 到 $y$ 结点最短路径上所有节点的值都加上 $z$。

- `2 x y`，表示求树从 $x$ 到 $y$ 结点最短路径上所有节点的值之和。

- `3 x z`，表示将以 $x$ 为根节点的子树内所有节点值都加上 $z$。

- `4 x` 表示求以 $x$ 为根节点的子树内所有节点值之和。

**【数据规模】**

 $1\le N \leq {10}^5$，$1\le M \leq {10}^5$。



--------

**考虑多次询问单次修改/多次修改单次询问：**

操作1：树上差分：a[$u$] += $z$ , a[$v$] += $z$, a[$LCA(u,v)$] -= $z$, a[fa[$LCA(u,v)$]] -= $z$

操作2：dfs预处理根结点到各点距离$dis$[], 求$dis$[u] + $dis$[v]- 2 * $dis$[$LCA(u,v)$]

操作3：树上差分：a[$x$]+=$z$

操作4：dfs处理子树和$sz$[]

---------

**如何高效处理对链多次询问多次修改呢？**

考虑将树按照一定的规则划分成数条链，链与链之间采用一定的方式进行索引，再按照处理区间的办法处理每条链上的查询与修改。



## 重链剖分

**重要概念：**

重儿子：父亲节点的所有儿子中**子树结点数目最多**（ $sz$ 最大）的结点；

轻儿子：父亲节点中除了重儿子以外的儿子；

重边：父亲结点和重儿子连成的边；轻边：父亲节点和轻儿子连成的边；

重链：由多条重边连接而成的路径；轻链：由多条轻边连接而成的路径；

**构造方法：**

1.跑一遍dfs，处理所有点的父节点数组 $fa$[] ，深度 $dep$[] ，子树大小 $sz$[] ，重儿子 $son$[] 

```c++
void dfs1(int u){
    // 初始化子树大小为1
	sz[u]=1;
	for(int i=head[u];i;i=nxt[i]){
		int v=edge[i].v;
        // 排除回退边
		if(v!=fa[u]){
			fa[v]=u;
			dep[v]=dep[u]+1;
            // 先dfs后计算子树大小
			dfs1(v);
			sz[u]+=sz[v];
            // 标记重儿子
			if(sz[v]>sz[son[u]])son[u]=v;
		}
	}
}
```

对于下图的树，规定红边连接的是重儿子

![](https://cdn.jsdelivr.net/gh/Beamstripe/img/2022/ex.png)

2.另跑一遍dfs，不同点在于此时是重儿子优先，并记录剖分后的dfs序 $dfn$[] ，同时记录 $dfn$[] 对原结点编号的映射 $ori$[]，确定重链顶点 $top$[]

```c++
void dfs2(int u,int tp){
    // 记录重链顶点为tp
	top[u]=tp;
    // 标记dfs序
	dfn[u]=++cnt;
    // dfs序映射原结点
	ori[cnt]=u;
    // 重儿子优先dfs
	if(son[u]){
        // 结点的重儿子的重链顶点与父结点的相同(tp)
		dfs2(son[u],tp);
        // 遍历轻儿子
		for(int i=head[u];i;i=nxt[i]){
			int v=edge[i].v;
            // 对轻链底端的结点其top是其本身
			if(v!=fa[u]&&v!=son[u]){
				dfs2(v,v);
			}
		}
	} 
}
```

上图的dfs序如下：

![](https://cdn.jsdelivr.net/gh/Beamstripe/img/2022/ex_dfn.png)

| 编号i         | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | 17   | 18   |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 点权$w$       | 0    | 0    | 1    | 1    | 0    | 1    | 0    | 1    | 1    | 1    | 0    | 1    | 1    | 1    | 1    | 0    | 1    | 0    |
| dfs序$dfn$    | 1    | 11   | 12   | 13   | 15   | 14   | 16   | 17   | 18   | 2    | 8    | 9    | 10   | 3    | 4    | 6    | 5    | 7    |
| 深度$dep$     | 0    | 1    | 2    | 3    | 4    | 4    | 2    | 3    | 4    | 1    | 2    | 3    | 4    | 2    | 3    | 4    | 4    | 3    |
| 重链顶点$top$ | 1    | 2    | 2    | 2    | 5    | 2    | 7    | 7    | 7    | 1    | 11   | 11   | 11   | 1    | 1    | 16   | 1    | 18   |

借助dfs序对树重新编号，可以将这棵树分成多条编号连续的重链与轻链，从而可以进行区间操作。

3.对 $dfn$[] 架上线段树，即先利用点权 $wgt$[]与 $ori$[] 建立线段树，之后的操作均在线段树上进行

```c++
void buildTree(int root,int l,int r){
	if(l==r){
		segTree[root].l=segTree[root].r=l;
		segTree[root].val=wgt[ori[l]];
		return;
	}
	int mid=(l+r)>>1;
	buildTree(root<<1,l,mid);
	buildTree(root<<1|1,mid+1,r);
	segTree[root].l=segTree[root<<1].l;
	segTree[root].r=segTree[root<<1|1].r;
	pushUp(root);
}
```

4.树上修改与查询操作：

**链操作**

基本思想：找到结点对应的 $dfn$[] 编号，在操作重链上相应结点后向上跳直到LCA

```c++
int queryChain(int u,int v){
	int ans=0;
    // 不在同一条重链时
	while(top[u]!=top[v]){
        // top深度大的优先向上跳
		if(dep[top[u]]>dep[top[v]])swap(u,v);
        // 处理同一条重链的询问
		ans+=getsum(1,dfn[top[v]],dfn[v]);
        // 向上跳至top的父结点
		v=fa[top[v]];
	}
    // 此时位于同一条重链上，直接处理询问
	if(dfn[u]>dfn[v])swap(u,v);
	ans+=getsum(1,dfn[u],dfn[v]);
	return ans;
}
```

**子树操作**

直接处理线段树区间 $dfn[x]$ 到 $dfn[x]+sz[x]-1$ 即可（为什么？）



## 复杂度

由于对轻边 $(u,v)$ （其中 $v=fa[u]$ ）有 $sz[v]<sz[u]/2$ ，且重链数小于 $\log n$ （证明略），得树链剖分的时间复杂度为 $O(n\log n)$



## 小技巧

1.在需要对多棵树进行树链剖分时，为避免超内存一般采用线段树动态开点的方法建树

2.需要对边权进行处理时，可将根结点的点权置0，边权赋值给子结点（深度较大的结点）



参考博客：https://www.luogu.com.cn/blog/communist/shu-lian-pou-fen-yang-xie

拓展知识：树形DP，[长链剖分](https://blog.csdn.net/litble/article/details/87965999)xxxxxxxxxx #pragma GCC optimize(3)#include<bits/stdc++.h>#define MAXN 1000005#define MAXM 10000005#define INF 1000000000#define MOD 1000000007#define F first#define S secondusing namespace std;typedef long long ll;typedef pair<int,int> P;int n,m,k,a[MAXN],b[MAXN];P save[2*MAXM];int pa[MAXM],pb[MAXM];int main(){    scanf("%d%d",&n,&m);    for(int i=1;i<=n;i++) scanf("%d",&a[i]);    for(int i=1;i<=m;i++) scanf("%d",&b[i]);    P p=P(0,0),q=P(0,0);    vector<int> va,vb;    memset(pa,0,sizeof(pa));    memset(pb,0,sizeof(pb));    for(int i=1;i<=n;i++)        if(pa[a[i]]) p=P(pa[a[i]],i); else {pa[a[i]]=i; va.push_back(i);}    for(int i=1;i<=m;i++)        if(pb[b[i]]) q=P(pb[b[i]],i); else {pb[b[i]]=i; vb.push_back(i);}    if(p.F!=0&&q.F!=0)    {        printf("%d %d %d %d\n",p.F,p.S,q.F,q.S);        return 0;    }    for(int i=1;i<=20000000;i++) save[i]=P(0,0);    for(int i=0;i<(int)va.size();i++)        for(int j=0;j<(int)vb.size();j++)        {            int sum=a[va[i]]+b[vb[j]];            if(save[sum].F)            {                printf("%d %d %d %d\n",save[sum].F,va[i],min(vb[j],save[sum].S),max(vb[j],save[sum].S));                return 0;            }            save[sum]=P(va[i],vb[j]);        }    puts("-1");    return 0;}c++
````