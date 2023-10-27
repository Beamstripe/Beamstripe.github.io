---
title: '2022 Summer ACM training-Newcoder Vol.3'
date: 2022-07-28
permalink: /posts/2022/07/nc3/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

# A [Ancestor](https://ac.nowcoder.com/acm/contest/33188/A)

题目大意：

给出两棵编号1-n的树$A$, $B$，$A$, $B$树上每个节点均有一个权值，给出$k$个关键点的编号$x_1…x_n$，问有多少种方案使得去掉恰好一个关键点使得剩余关键点在树$A$上$LCA$的权值大于树$B$上$LCA$的权值。

多个点的共同$LCA$实际上只需要将$DFS$序最小与$DFS$序最大的点求一个$LCA$就行

那么就能利用set/multiset来动态维护关键点数组$DFS$序的最大/最小值，每次取出一个点判断就行

```c++
#include <bits/stdc++.h>
struct Tree {
    std::vector<int> sz, top, dep, parent, in;
    int cur;
    std::vector<std::vector<int> > e;
    Tree(int n) : sz(n), top(n), dep(n), parent(n, -1), e(n), in(n), cur(1) {}
    void addEdge(int u, int v) {
        e[u].push_back(v);
        e[v].push_back(u);
    }
    void init() {
        dfsSz(1);
        dfsHLD(1);
    }
    void dfsSz(int u) {
        if (parent[u] != -1)
            e[u].erase(std::find(e[u].begin(), e[u].end(), parent[u]));
        sz[u] = 1;
        for (int &v : e[u]) {
            parent[v] = u;
            dep[v] = dep[u] + 1;
            dfsSz(v);
            sz[u] += sz[v];
            if (sz[v] > sz[e[u][0]])
                std::swap(v, e[u][0]);
        }
    }
    void dfsHLD(int u) {
        in[u] = cur++;
        for (int v : e[u]) {
            if (v == e[u][0]) {
                top[v] = top[u];
            } else {
                top[v] = v;
            }
            dfsHLD(v);
        }
    }
    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                u = parent[top[u]];
            } else {
                v = parent[top[v]];
            }
        }
        if (dep[u] < dep[v]) {
            return u;
        } else {
            return v;
        }
    }
};
const int MAXN =100005;
using namespace std;
int key[MAXN],w1[MAXN],w2[MAXN];
int main(){
	int n,k;
	cin>>n>>k;
	Tree ta(n+1),tb(n+1);
	for(int i=1;i<=k;i++){
		cin>>key[i];
	}
	for(int i=1;i<=n;i++){
		cin>>w1[i];
	}
	for(int i=2;i<=n;i++){
		int x;
		cin>>x;
		ta.addEdge(i,x);
	}
	for(int i=1;i<=n;i++){
		cin>>w2[i];
	}
	for(int i=2;i<=n;i++){
		int x;
		cin>>x;
		tb.addEdge(i,x);
	}
	ta.init();
	tb.init();
	set<int, function<bool(int, int)>> sta([&](int i, int j) {
        return ta.in[i] < ta.in[j];
    });
    set<int, function<bool(int, int)>> stb([&](int i, int j) {
        return tb.in[i] < tb.in[j];
    });
	for(int i=1;i<=k;i++){
		sta.insert(key[i]);
		stb.insert(key[i]);
	}
	int ans=0;
	for(int i=1;i<=k;i++){
		sta.erase(key[i]);
		stb.erase(key[i]);
		int lca_a=ta.lca(*sta.begin(),*sta.rbegin());
		int lca_b=tb.lca(*stb.begin(),*stb.rbegin());
		if(w1[lca_a]>w2[lca_b])++ans;
		sta.insert(key[i]);
		stb.insert(key[i]);
	}
	cout<<ans<<endl;
	return 0;
} 
```

# B [Boss](https://ac.nowcoder.com/acm/contest/33188/B)

题目大意：将n（1e5）分到k (10) 个城市中，配额确定，求最小花费

一个比较朴素的想法就是最小费用最大流，但是复杂度明显是不对的，点数1e5，边数高达1e6。

但是我们注意到，增广时，每次的流量只有1，因为我们把N个人都看成是不同的。同时这个图只有两层，将一个已经分配好的人换到另一个城市就意味着走一条反向边，再走一条正向边，我们发现这个过程就是复杂度的瓶颈。所以我们考虑把这个繁琐的过程简化，用一条边直接代表正向边+反向边。



这样我们的图上一共建立K个点代表K个城市，我们直接把N个人先分配到第1个城市，这样其它城市再从第1个城市抢人。对于每个人，我可以从当前所在城市变换到其它任意城市，代价就是两者之差，所以一共要建K条边。



这样我们就发现，图上有很多代价不同的重边。用 $K^2$ 个堆维护两点之间的重边，跑SPFA（最小费用），每转移一个人重新维护一次边表。

复杂度 $O(N * SPFA(K^2) + N K \log(N))$

std:

```c++
#include<bits/stdc++.h>
using namespace std;

int e[15], c[101010][15];

priority_queue<pair<int, int> > edge[15][15]; 

int belong[101010];
int dis[15], frm[15], inq[15];

int spfa(int s)
{
    queue<int> Q;
    Q.push(s);
    for(int i = 0; i < 10; i++)
    {
        dis[i] = 1e9;
        inq[i] = 0;
        frm[i] = 0;
    }
    dis[s] = 0;
    inq[s] = 1;
    while(!Q.empty())
    {
        int u = Q.front();
        Q.pop();
        inq[u] = 0;
        for(int v = s - 1; v >= 0; v--)
        {
            if(edge[u][v].empty())
                continue;
            int d = -edge[u][v].top().first;
            if(dis[v] > dis[u] + d)
            {
                dis[v] = dis[u] + d;
                frm[v] = u;
                if(!inq[v])
                {
                    inq[v] = 1;
                    Q.push(v);
                }
            }
        }
    }
    
    vector<int> moved;
    int t = 0;
    while(t != s)
    {
        moved.push_back(edge[frm[t]][t].top().second);
        belong[edge[frm[t]][t].top().second] = frm[t];
        t = frm[t];
    }

    for(int u: moved)
    {
        for(int i = 0; i <= s; i++)
        if(i != u)
            edge[i][belong[u]].push(make_pair(-c[u][i] + c[u][belong[u]], u));
    }

    return dis[0];
}

int main()
{
    int n, k;
    scanf("%d%d", &n, &k);
    for(int i = 0; i < k; i++)
        scanf("%d", &e[i]);
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < k; j++)
            scanf("%d", &c[i][j]);
    
    long long ans = 0;

    for(int i = 0; i < n; i++)
    {
        ans += c[i][0];
        belong[i] = 0;
    }
    
    for(int j = 1; j < k; j++)
    {
        for(int i = 0; i < n; i++)
            edge[j][belong[i]].push(make_pair(-c[i][j] + c[i][belong[i]], i));
        for(int i = 0; i < e[j]; i++)
        {
            for(int u = 0; u <= j; u++)
                for(int v = 0; v <= j; v++)
                    while(!edge[u][v].empty() && belong[edge[u][v].top().second] != v)
                        edge[u][v].pop();
            ans += spfa(j);
        }
    }
    printf("%lld\n", ans);
    return 0;
}
```

# C [Concatenation](https://ac.nowcoder.com/acm/contest/33188/C)

题目大意：给定n个字符串，求一个将他们拼接起来的方案，使得结果的字典序最小。

时限4s：签到题

基本思想：

```
sort(v.begin(),v.end(),[&](string a, string b) {
        return a+b<b+a;
    });
```

暴力-WA：

```c++
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

vector<string> v;

int main(){
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        string str;
        cin>>str;
        v.push_back(str);
    }
    sort(v.begin(),v.end(),[&](string a, string b) {
        return a+b<b+a;
    });
    for(int i=0;i<n;i++){
        cout<<v[i];
    }
    cout<<endl;
    return 0;
}
```

赛时用了结构体擦边3787ms AC

实际上std是需要进行一定优化的（可满足3s时限）

将n个字符串建一个trie。对于两个字符串a, b，不妨设|a| < |b|。如果a不是b的前缀，则可以直接比较二者在dfs序的大小；否则，可以用Z-algorithm(也称扩展KMP)来判断ab和ba的大小关系。

标程：

```c++
#include<bits/stdc++.h>
 
using namespace std;
 
const int N = 2000005;
const int sigma = 5;
const int sumS = 20000005;
 
int n, cnt, tr[sumS][sigma];
int endcnt[sumS], sid[sumS];
vector<bool> is_lower[N];
string s[N];
 
int dfs_u[sumS], dfs_top, stk[sumS], stk_top;
char dfs_i[sumS]; 
vector<bool> vis;
int pre[sumS], suf[sumS], head, tail;
void dfs() {
    dfs_u[++dfs_top] = 0;
    dfs_i[dfs_top] = 0;
    while (dfs_top) {
        int u = dfs_u[dfs_top];
        if (dfs_i[dfs_top] == 0) {
            if (u) stk[stk_top++] = u;
 
            if (endcnt[u]) {
                vector<int> ids;
                for (int i = 0; i < stk_top; i++) if (endcnt[stk[i]] && is_lower[sid[u]][i]) {
                    vis[stk[i]] = true;
                    ids.push_back(stk[i]);
                }
                int pos = tail;
                for (int i : ids) if (!vis[pre[i]]) {
                    pos = i;
                    break;
                }
                for (int i : ids) vis[i] = false;
                int p = pre[pos];
                pre[pos] = suf[p] = u;
                pre[u] = p, suf[u] = pos;
            }
        }
        while (!tr[u][dfs_i[dfs_top]] && dfs_i[dfs_top] < sigma) {
            dfs_i[dfs_top]++;
        }
 
        if (dfs_i[dfs_top] == sigma) {
            if (u) stk_top--;
            dfs_top--;
        } else {
            int x = tr[u][dfs_i[dfs_top]];
            dfs_i[dfs_top]++;
            dfs_top++;
            dfs_u[dfs_top] = x;
            dfs_i[dfs_top] = 0;
        }
        
    }
}
int main() {
	freopen("test.txt","r",stdin);
	freopen("test.out","w",stdout);
    ios::sync_with_stdio(false), cin.tie(0);
    vis.resize(sumS);
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> s[i];
 
    for (int i = 1; i <= n; i++) {
        int len = s[i].size();
        int l = 0, r = 0;
        is_lower[i].resize(len);
 
        vector<int> lcp(len + 1);
        lcp[0] = len;
        for (int j = 1; j <= len; j++) {
            lcp[j] = j > r ? 0 : min(r - j + 1, lcp[j - l]);
 
            while (j + lcp[j] < len && s[i][lcp[j]] == s[i][j + lcp[j]])
                lcp[j]++;
            
            if (j + lcp[j] - 1 > r) {
                l = j;
                r = j + lcp[j] - 1;
            }
        }
 
        for (int j = 1; j < len; j++) {
            int tmp = lcp[j];
            if (tmp == len - j) {
                tmp = lcp[len - j];
                if (tmp == j) is_lower[i][j - 1] = false;
                else {
                    int posl = tmp + len - j;
                    int posr = tmp;
                    is_lower[i][j - 1] = s[i][posr] < s[i][posl];
                }
            } else {
                int posl = tmp;
                int posr = tmp + j;
                is_lower[i][j - 1] = s[i][posr] < s[i][posl];
            }
        }
        is_lower[i][len - 1] = false;
    }
 
    cnt = 0;
    for (int i = 1; i <= n; i++) {
        int p = 0;
        for (char c : s[i]) {
            if (!tr[p][c - '0'])
                tr[p][c - '0'] = ++cnt;
            p = tr[p][c - '0'];
        }
        sid[p] = i;
        endcnt[p]++;
    }
 
    head = cnt + 1, tail = cnt + 2;
    pre[tail] = head, suf[head] = tail;
    pre[head] = -1, suf[tail] = -1;
    
    dfs();
 
    for (int p = suf[head]; p != tail; p = suf[p]) {
        for (int tt = 0; tt < endcnt[p]; tt++) {
            cout << s[sid[p]];
        }
    }
}
```

# D [Fief](https://ac.nowcoder.com/acm/contest/33188/F)

题意：给定一棵树和一个起点，1号节点为终点，随机选其中K条边变成指向终点的单向边，在树上随机游走，求到达终点的期望步数

我们知道，从任意一点出发，随机选择一条边移动，在树上移动到其父亲的期望步数为2 * 子树大小 -1，移动到根，就是每一条边的期望步数之和。

之后考虑单向边的影响，单向边对于其下方的子树没有任何影响，但是会使上方的节点的子树大小减少（因为是单向边，不能走到这里，相当于没有这个子树）。

那么我们考虑每个单向边的贡献，如果一条单向边与上方的边之间的没有其它单向边，全都是双向的，那么那么这条单向边对于期望的贡献就是 -2*子树大小。那么我们也要计算产生这种情况的概率作为系数，发现这个概率只与两条边之间的距离有关，所以我们只需某一距离有多少对边即可。发现一条边可以产生的贡献的距离是连续的，所以直接用区间加法统计即可。

```c++
#include<bits/stdc++.h>
using namespace std;

const int mod = 998244353;

int fac[1010101], inv_fac[1010101];

int qpow(int x, int k)
{
    int ret = 1;
    while(k)
    {
        if(k & 1)
            ret = 1ll * ret * x % mod;
        x = 1ll * x * x % mod;
        k /= 2;
    }
    return ret;
}

int inv(int x)
{
    return qpow(x, mod - 2);
}

int C(int n, int m)
{
    if(n < 0)
        return 0;
    if(0 > m || m > n)
        return 0;
    return 1ll * fac[n] * inv_fac[m] % mod * inv_fac[n - m] % mod;
}

int fa[1010101];

int depth[1010101], sz[1010101], un[1010101];
int cnt[1010101];

struct Edge{
    int to, nxt;
}e[2020202];

int head[1010101], edge_cnt = 0;

void dfs(int u)
{
    sz[u] = 1;
    for(int i = head[u]; i > 0; i = e[i].nxt)
    {
        int v = e[i].to;
        if(v == fa[u])
            continue;
        fa[v] = u;
        depth[v] = depth[u] + 1;
        dfs(v);
        sz[u] += sz[v];
    }
}

void add(int u, int v)
{
    edge_cnt++;
    e[edge_cnt].to = v;
    e[edge_cnt].nxt = head[u];
    head[u] = edge_cnt;
}


int get_un(int u)
{
    if(un[u] == -1)
        un[u] = get_un(fa[u]);
    return un[u];
}

int main()
{
    const int MAX_NUM = 1e6;
    fac[0] = 1;
    for(int i = 1; i <= MAX_NUM; i++)
        fac[i] = 1ll * fac[i - 1] * i % mod;
    inv_fac[MAX_NUM] = inv(fac[MAX_NUM]);
    for(int i = MAX_NUM; i >= 1; i--)
        inv_fac[i - 1] = 1ll * inv_fac[i] * i % mod;
    int N, K, s;
    scanf("%d%d%d", &N, &K, &s);

    for(int i = 1; i < N; i++)
    {
        int u, v;
        scanf("%d%d", &u, &v);
        add(u, v);
        add(v, u);
    }
    fa[1] = 0;
    depth[1] = 0;
    dfs(1);

    for(int i = 1; i <= N; i++)
        un[i] = -1;

    int ans = 0;
    while(s != 1)
    {
        ans += 2 * sz[s] - 1;
        ans %= mod;
        un[s] = depth[s];
        s = fa[s];
    }
    un[1] = 0;

    for(int i = 2; i <= N; i++)
    {
        cnt[depth[i]] -= sz[i];
        if(cnt[depth[i]] < 0)
            cnt[depth[i]] += mod;

        int un_i = get_un(i);
        if(depth[i] > un_i)
        {
            cnt[depth[i] - un_i] += sz[i];
            if(cnt[depth[i] - un_i] >= mod)
                cnt[depth[i] - un_i] -= mod;
        }
        else if(depth[i] == un_i)
        {
            cnt[1] += sz[i];
            if(cnt[1] >= mod)
                cnt[1] %= mod;
        }
    }
    for(int i = 1; i <= N; i++)
    {
        cnt[i] += cnt[i - 1];
        if(cnt[i] >= mod)
            cnt[i] -= mod;
    }
    assert(cnt[0] == 0);
    assert(cnt[N] == 0);

    int sum = 0;
    for(int i = 1; i <= N; i++)
    {
        sum += 1ll * cnt[i] * C(N - 1 - i, K - 1) % mod;
        if(sum >= mod)
            sum -= mod;
    }
    ans = ans - 2ll * sum * inv(C(N - 1, K)) % mod;
    if(ans < 0)
        ans += mod;
    printf("%d\n", ans);
    return 0;
}
```

# H [Hacker](https://ac.nowcoder.com/acm/contest/33188/H)

给出长度为n的小写字符串A和k个长度为m的小写字符串 $B_1…B_k$，B的每个位置拥有统一的权值$v_1…v_m$，对于每个$B_i$求最大和区间满足该区间构成的字符串是A的子串（空区间合法）。

我们可以将问题进行转化，相当于对$B_i$的每个位置求出它作为结束位置在$A$中的最长子串长度，然后在该区间求最大子段和，所有位置的最大值即为答案。对于每个位置的最长子串，可以对A建后缀自动机，然后$B_i$从左往右在A的后缀自动机上转移，如果当前节点无法转移跳至父亲节点，最后无法转移则长度为0，转移成功则为转移前节点的最大长度+1。

# J [Journey](https://ac.nowcoder.com/acm/contest/33188/J)

阅读理解题：建议配合样例食用

题目大意：给定一个城市有若干十字路口，右转需要等红灯，直行、左转和掉头都需要，求起点到终点最少等几次红灯

初步想法：

把每条道路看成两个点，十字路口连边，十字路口右转的边边权为0，直行、左转边权为1，掉头为两条是边权为1相反的边，如图

![image-20220727203608588](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220727203608588.png)

但在 $n\le500000$ 下该建图方式不可取

考虑采取类似的思维直接模拟，SPFA最短路

标程：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
typedef tuple<int, int, int> tup;
const int N = 1000005;
int n, m, cross[N][4], dis[N][4];
bool vis[N][4];
unordered_map<int, int> id[N];
int st1, st2, en1, en2;
int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < 4; j++) {
            cin >> cross[i][j];
            if (cross[i][j]) id[i][cross[i][j]] = j;
        }
    }
    cin >> st1 >> st2 >> en1 >> en2;
    memset(dis, 0x3f, sizeof(dis));
    int idx = id[st2][st1];
    dis[st2][idx] = 0;
    priority_queue<tup, vector<tup>, greater<tup>> PQ;
    PQ.push({0, st2, idx});
    while (!PQ.empty()) {
        auto [d, u, x] = PQ.top();
        PQ.pop();
        if (vis[u][x]) continue;
        vis[u][x] = true;
        for (int i = 0; i < 4; i++) {
            int delta = (i != (x + 1) % 4);
            int v = cross[u][i];
            if (!v) continue;
            int j = id[v][u];
            if (d + delta < dis[v][j]) {
                dis[v][j] = d + delta;
                PQ.push({dis[v][j], v, j});
            }
        }
    }
    int ans = dis[en2][id[en2][en1]];
    if (ans > n * 4) {
        ans = -1;
    }
    cout << ans; 
}
```