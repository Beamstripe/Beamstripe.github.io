---
title: '2022 Summer ACM training-Newcoder Vol.8'
date: 2022-08-13
permalink: /posts/2022/08/2022SummerTraining16-Newcoder8/
tags:
  - Chinese post
  - ACM
  - Algorithm
---

![image-20220813181013731](https://cdn.jsdelivr.net/gh/Beamstripe/img/img/2022/image-20220813181013731.png)

# D [Poker Game: Decision](https://ac.nowcoder.com/acm/contest/33193/D)

暴力搜索+减枝

```c++
#include <bits/stdc++.h>
using namespace std;
struct card{
  char suit;
  int rank;
  card(){
  }
  bool operator <(card C){
    return rank < C.rank || rank == C.rank && suit < C.suit;
  }
};
istream& operator >>(istream& is, card& C){
  string S;
  is >> S;
  if (S[0] == 'A'){
    C.rank = 14;
  } else if (S[0] == 'K'){
    C.rank = 13;
  } else if (S[0] == 'Q'){
    C.rank = 12;
  } else if (S[0] == 'J'){
    C.rank = 11;
  } else if (S[0] == 'T'){
    C.rank = 10;
  } else {
    C.rank = S[0] - '0';
  }
  C.suit = S[1];
  return is;
}
vector<int> hand(vector<card> C){
  sort(C.begin(), C.end());
  set<char> suits;
  for (int i = 0; i < 5; i++){
    suits.insert(C[i].suit);
  }
  if (suits.size() == 1 && C[4].rank - C[0].rank == 4){
    if (C[4].rank == 14){
      return vector<int>{9};
    } else {
      return vector<int>{8, C[4].rank};
    }
  }
  if (suits.size() == 1 && C[3].rank == 5 && C[4].rank == 14){
    return vector<int>{8, 5};
  }
  if (C[0].rank == C[3].rank){
    return vector<int>{7, C[0].rank, C[4].rank};
  }
  if (C[1].rank == C[4].rank){
    return vector<int>{7, C[1].rank, C[0].rank};
  }
  if (C[0].rank == C[2].rank && C[3].rank == C[4].rank){
    return vector<int>{6, C[0].rank, C[3].rank};
  }
  if (C[2].rank == C[4].rank && C[0].rank == C[1].rank){
    return vector<int>{6, C[2].rank, C[0].rank};
  }
  if (suits.size() == 1){
    return vector<int>{5, C[4].rank, C[3].rank, C[2].rank, C[1].rank, C[0].rank};
  }
  if (C[1].rank - C[0].rank == 1 && C[2].rank - C[1].rank == 1 && C[3].rank - C[2].rank == 1 && C[4].rank - C[3].rank == 1){
    return vector<int>{4, C[4].rank};
  }
  if (C[0].rank == 2 && C[1].rank == 3 && C[2].rank == 4 && C[3].rank == 5 && C[4].rank == 14){
    return vector<int>{4, 5};
  }
  if (C[0].rank == C[2].rank){
    return vector<int>{3, C[0].rank, C[4].rank, C[3].rank};
  }
  if (C[1].rank == C[3].rank){
    return vector<int>{3, C[1].rank, C[4].rank, C[0].rank};
  }
  if (C[2].rank == C[4].rank){
    return vector<int>{3, C[2].rank, C[1].rank, C[0].rank};
  }
  if (C[0].rank == C[1].rank && C[2].rank == C[3].rank){
    return vector<int>{2, C[2].rank, C[0].rank, C[4].rank};
  }
  if (C[0].rank == C[1].rank && C[3].rank == C[4].rank){
    return vector<int>{2, C[3].rank, C[0].rank, C[2].rank};
  }
  if (C[1].rank == C[2].rank && C[3].rank == C[4].rank){
    return vector<int>{2, C[3].rank, C[1].rank, C[0].rank};
  }
  if (C[0].rank == C[1].rank){
    return vector<int>{1, C[0].rank, C[4].rank, C[3].rank, C[2].rank};
  }
  if (C[1].rank == C[2].rank){
    return vector<int>{1, C[1].rank, C[4].rank, C[3].rank, C[0].rank};
  }
  if (C[2].rank == C[3].rank){
    return vector<int>{1, C[2].rank, C[4].rank, C[1].rank, C[0].rank};
  }
  if (C[3].rank == C[4].rank){
    return vector<int>{1, C[3].rank, C[2].rank, C[1].rank, C[0].rank};
  }
  return vector<int>{0, C[4].rank, C[3].rank, C[2].rank, C[1].rank, C[0].rank};
}
int dfs(vector<vector<int>> &alice, vector<vector<int>> &bob, int a, int b, int p){
  if (p == 6){
    if (alice[a] > bob[b]){
      return 1;
    } else if (alice[a] < bob[b]){
      return -1;
    } else {
      return 0;
    }
  } else {
    int mx = -1;
    for (int i = 0; i < 6; i++){
      if ((a >> i & 1) == 0 && (b >> i & 1) == 0){
        if (p % 2 == 0){
          mx = max(mx, -dfs(alice, bob, a | (1 << i), b, p + 1));
        } else {
          mx = max(mx, -dfs(alice, bob, a, b | (1 << i), p + 1));
        }
      }
    }
    return mx;
  }
}
int main(){
  int T;
  cin >> T;
  for (int i = 0; i < T; i++){
    vector<card> a(2);
    cin >> a[0] >> a[1];
    vector<card> b(2);
    cin >> b[0] >> b[1];
    vector<card> c(6);
    cin >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5];
    vector<vector<int>> alice(1 << 6), bob(1 << 6);
    for (int j = 0; j < (1 << 6); j++){
      if (__builtin_popcount(j) == 3){
        vector<card> ha = {a[0], a[1]};
        vector<card> hb = {b[0], b[1]};
        for (int k = 0; k < 6; k++){
          if ((j >> k & 1) == 1){
            ha.push_back(c[k]);
            hb.push_back(c[k]);
          }
        }
        alice[j] = hand(ha);
        bob[j] = hand(hb);
      }
    }
    int ans = dfs(alice, bob, 0, 0, 0);
    if (ans == 1){
      cout << "Alice" << endl;
    } else if (ans == -1){
      cout << "Bob" << endl;
    } else {
      cout << "Draw" << endl;
    }
  }
}
```

# F [ Longest Common Subsequence](https://ac.nowcoder.com/acm/contest/33193/F)

签到题：

对于数列$x_i$，若满足$x_{i+1}=ax_i^2+bx_i+c\mod p$，一定存在一个周期 $T$  ，起始位置 $s$ ，当 $i\ge s$ 时， $x_{i+T}=x_{i}$ 

因此可以通过寻找生成数列 $x_i$中 $s$ 在两个数组中的位置确定LCS

赛时用unordered_map卡过了，std如下

```c++
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    for (cin >> T; T; T -= 1) {
        int n, m, p, x, a, b, c;
        cin >> n >> m >> p >> x >> a >> b >> c;
        auto next = [&](){
            x = ((LL)a * x % p * x + (LL)b * x + c) % p;
            return x;
        };
        vector<int> s(n), t(m);
        for (int& si : s) si = next();
        for (int& ti : t) ti = next();
        int cyc = -1, sp = 0;
        for (int i = 0; i < m; i += 1)
            if (t[i] == s.back()) {
                cyc = i + 1;
                break;
            }
        if (cyc == -1) {
            for (int i = n - 1; i >= 0; i -= 1)
                if (t.back() == s[i]) {
                    cyc = m + n - i - 1;
                    break;
                }
        }
        if (cyc != -1) {
            int i = n - 1, j = cyc - 1;
            while (i >= 0 and j >= -n) {
                int x = s[i], y = j >= 0 ? (j >= m ? x : t[j]) : s[n + j];
                if (x == y) {
                    i -= 1;
                    j -= 1;
                }
                else break;
            }
            i += 1;
            j += 1;
            j = (j % cyc + cyc) % cyc;
            int ans = 0;
            while (i < n) {
                ans = max(ans, min(n - i, m - j));
                i += 1;
                j += 1;
                if (j == cyc) j = 0;
            }
            cout << ans << "\n";
            continue;
        }
        else cout << "0\n";
    }
}
```

# G [Lexicographic Comparison](https://ac.nowcoder.com/acm/contest/33193/G)

暴力：

```
#include<iostream>
#include<cstring>
#include<algorithm>
#include<vector>
#include<queue>
#include<map>
#include<vector>
#include<cmath>
#include<set>
#include<queue>

using namespace std;
#define IOS ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
#define int long long
#define endl "\n"
#define PII pair<int,int>
#define PVI pair<vector<int>*, int>
#define rep(i, x, y) for (int i = x; i <= y; i++)
#define l first
#define r second

const int N = 1e5+10, M = 2*N, inf = 0x3f3f3f3f;

int from[N], to[N], a[N];
int tf[N];
void solve()
{
	int n, q;
	cin >> n >> q;
	for (int i = 1; i <= n; i++) a[i] = i, from[i] = i;
	string opt;
	int aa, bb;
	while (q--){
		cin >> opt;
		cin >> aa >> bb;
		if (opt == "swap_a"){
			swap(a[aa], a[bb]);
		}
		else if (opt == "swap_p"){
			swap(from[aa], from[bb]);
		}
		else{
			if (aa == bb) {
				cout << '=' << endl;
				continue;
			}
			int flag = 1;
			for (int i = 1; i <= n; i++){
				if (from[i] != i){
					int m = i, ta = 0, tb = 0;
					int ka = aa-1, kb = bb-1;
					int k = 0;
					tf[k] = m;
					if (ka == 0) ta = m;
					if (kb == 0) tb = m;
					while (from[m] != i){
						m = from[m];
						k++;
						tf[k] = m;
						if (k == ka) ta = m;
						if (k == kb) tb = m;
						if (ta&&tb) break;
					}
					tf[++k] = i;
					if (!ta || !tb) ta = tf[ka%k], tb = tf[kb%k];
					if (a[ta] < a[tb]){
						cout << '<' << endl;
						flag = 0;
						break;
					}
					else if (a[ta] > a[tb]){
						cout << '>' << endl;
						flag = 0;
						break;
					}
				}
			}
			if (flag) cout << '=' << endl;
		}
	}
}

signed main() 
{
    IOS
    int _t = 1;
	cin >> _t;
	while(_t--) solve();
	return 0;

}
```

正解

维护 p形成的环.
只需要比较环长不整除 $x − y$ 的环中涉及的最小下标.
平衡树维护所有环.
方法一: 维护所有环长的对应最小下标, 只有 $\sqrt q$种环长.
方法二: 维护区间最小公倍数.
复杂度 $O(q\sqrt q)$ 或 $O(wq\log q)$ ($w = 64$).

```c++
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
struct Node{
    int v, p, size, mv;
    Node *L, *R, *par;
    Node(int v, int p, Node* L = nullptr, Node* R = nullptr) : v(v), p(p), size(1), mv(v), L(L), R(R), par(nullptr) {}
    Node* copy(Node* L, Node* R) {
        this->L = L;
        this->R = R;
        return this;
    }
};
namespace Treap {
    mt19937 rng;
    Node* update(Node* p) {
        p->size = 1;
        p->mv = p->v;
        if (p->L) p->size += p->L->size, p->mv = min(p->mv, p->L->mv), p->L->par = p;
        if (p->R) p->size += p->R->size, p->mv = min(p->mv, p->R->mv), p->R->par = p;
        p->par = nullptr;
        return p;
    }
    Node* merge(Node*L, Node* R) {
        if (not L) return R;
        if (not R) return L;
        if (L->p < R->p) return update(L->copy(L->L, merge(L->R, R)));
        return update(R->copy(merge(L, R->L), R->R));
    }
    pair<Node*, Node*> split_rank(Node* p, int r) {
        if (not p) return {};
        int Lsize = p->L ? p->L->size : 0;
        if (Lsize + 1 <= r) {
            auto [L, R] = split_rank(p->R, r - Lsize - 1);
            return {update(p->copy(p->L, L)), R};
        }
        auto [L, R] = split_rank(p->L, r);
        return {L, update(p->copy(R, p->R))};
    }
    Node* find_root(Node* p) {
        while (p->par)
            p = p->par;
        return p;
    }
    int rank(Node* p) {
        int res = 1;
        if (p->L) res += p->L->size;
        while (p->par) {
            if (p->par->R == p) {
                res += 1;
                if (p->par->L) res += p->par->L->size;
            }
            p = p->par;
        }
        return res;
    }
    Node* kth(Node* p, int k) {
        int Lsize = p->L ? p->L->size : 0;
        if (Lsize + 1 == k) return p;
        if (k <= Lsize) return kth(p->L, k);
        return kth(p->R, k - Lsize - 1);
    }
};
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    for (cin >> T; T; T -= 1) {
        int n, q;
        cin >> n >> q;
        map<int, int> a;
        map<int, Node*> p;
        map<int, set<int>> ms;
        for (int qi = 0; qi < q; qi += 1) {
            string op;
            LL x, y;
            cin >> op >> x >> y;
            if (op == "swap_a" and x != y) {
                if (not a.count(x)) a[x] = x;
                if (not a.count(y)) a[y] = y;
                swap(a[x], a[y]);
            }
            if (op == "swap_p" and x != y) {
                if (not p.count(x)) {
                    p[x] = new Node(x, Treap::rng());
                    ms[1].insert(x);
                }
                if (not p.count(y)) {
                    p[y] = new Node(y, Treap::rng());
                    ms[1].insert(y);
                }
                Node *rtx = Treap::find_root(p[x]), *rty = Treap::find_root(p[y]);
                auto rx = Treap::rank(p[x]);
                auto ry = Treap::rank(p[y]);
                if (rtx == rty) {
                    ms[rtx->size].erase(rtx->mv);
                    if (ms[rtx->size].empty())
                        ms.erase(rtx->size);
                    if (rx > ry) {
                        swap(rx, ry);
                        swap(x, y);
                    }
                    auto [L, R] = Treap::split_rank(rtx, rx);
                    auto [RL, RR] = Treap::split_rank(R, ry - rx);
                    auto rt = Treap::merge(L, RR);
                    ms[rt->size].insert(rt->mv);
                    ms[RL->size].insert(RL->mv);
                }
                else {
                    ms[rtx->size].erase(rtx->mv);
                    if (ms[rtx->size].empty())
                        ms.erase(rtx->size);
                    ms[rty->size].erase(rty->mv);
                    if (ms[rty->size].empty())
                        ms.erase(rty->size);
                    auto [Lx, Rx] = Treap::split_rank(rtx, rx);
                    auto [Ly, Ry] = Treap::split_rank(rty, ry);
                    auto rt = Treap::merge(Treap::merge(Lx, Ry), Treap::merge(Ly, Rx));
                    ms[rt->size].insert(rt->mv);
                }
            }
            if (op == "cmp") {
                int cmp = 0;
                int k = n + 1;
                for (auto& [c, s] : ms)
                    if ((y - x) % c)
                        k = min(k, *s.begin());
                if (k <= n) {
                    auto rt = Treap::find_root(p[k]);
                    int rk = Treap::rank(p[k]);
                    int rx = (rk - 1 + x - 1) % rt->size + 1;
                    int ry = (rk - 1 + y - 1) % rt->size + 1;
                    int px = Treap::kth(rt, rx)->v;
                    int py = Treap::kth(rt, ry)->v;
                    if (not a.count(px)) a[px] = px;
                    if (not a.count(py)) a[py] = py;
                    if (a[px] < a[py]) cmp = -1;
                    else cmp = 1;
                }
                if (cmp == 1) cout << ">\n";
                else if (cmp == -1) cout << "<\n";
                else cout << "=\n";
            }
        }
    }
}
```