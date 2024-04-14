---
title: codetop中等(5)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Codetop
excerpt: codetop中等(5)
math: true
date: 2024-04-10 21:45:00
---

## [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/description/)

- 递归 或 类似前序遍历的迭代

```C++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        if (root == nullptr) return res;

        stack<TreeNode*> st;

        st.push(root);
        while (!st.empty())
        {
            TreeNode* cur = st.top(); st.pop();
            res.push_back(cur->val);

            if (cur->left) st.push(cur->left);
            if (cur->right) st.push(cur->right);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```


## [50. Pow(x, n)](https://leetcode.cn/problems/powx-n/description/)

- 快速幂

```C++
class Solution {
public:
    double quickMul(double x, long n)
    {
        if (n == 0) return 1.0;

        double y = quickMul(x, n / 2);
        return n % 2 ? y * y * x : y * y;
    }

    double myPow(double x, int n) {
        long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }
};
```


## [450. 删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/description/)

- 根据二叉搜索树性质，如果删除的是叶子结点，则返回 nullptr，如果具有一个子节点，就返回子节点，如果有两个，则把左节点移动到右节点的最左边

```C++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root;

        // 如果当前节点是待删除的节点
        if (root->val == key)
        {
            // 叶子结点
            if (root->left == nullptr && root->right == nullptr) return nullptr;
            // 有一个子节点
            else if (root->left == nullptr) return root->right;
            else if (root->right == nullptr) return root->left;
            // 有两个，把 root 的左子树挂在右子树的最左边
            else
            {
                // 找到最左边节点
                TreeNode* cur = root->right;
                while (cur->left)
                {
                    cur = cur->left;
                }

                // 将左子树挂载
                cur->left = root->left;
                // 返回右子树
                return root->right;;
            }
        }

        // 子节点的处理
        if (root->val > key) root->left = deleteNode(root->left, key);
        if (root->val < key) root->right = deleteNode(root->right, key);

        return root;
    }
};
```


## [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/description/)

- 完全背包，组合问题，先遍历物品，再遍历背包

```C++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1, 0);

        dp[0] = 1;

        // 组合数
        // 先物品
        for (int i = 0 ; i < coins.size(); ++i)
        {
            // 后背包
            for (int j = coins[i]; j <= amount; ++j)
            {
                // 方案数，累和
                dp[j] += dp[j - coins[i]];
            }
        }

        return dp[amount];
    }
};
```


## [59. 螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/description/)

- 模拟

```C++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        // 边界
        int top = 0, down = n - 1;
        int left = 0, right = n - 1;

        // 存储结果
        vector<vector<int>> res(n, vector<int>(n));
        int count = 1;  // 用于计数
        while (1)
        {
            for (int i = left; i <= right; ++i)
            {   // 从左向右遍历
                res[top][i] = count++;
            }
            ++top;
            if (top > down) break;

            for (int i = top; i <= down; ++i)
            {   // 从上向下遍历
                res[i][right] = count++;
            }
            --right;
            if (left > right) break;

            for (int i = right; i >= left; --i)
            {   // 从右到左遍历
                res[down][i] = count++;
            }
            --down;
            if (top > down) break;

            for (int i = down; i >= top; --i)
            {   // 从下到上遍历
                res[i][left] = count++;
            }
            ++left;
            if (left > right) break;
        }

        return res;
    }
};
```



## [LCR 143. 子结构判断](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/description/)

- 双根递归

```C++
class Solution {
public:
    bool dfs(TreeNode* A, TreeNode* B)
    {
        // 如果 B 为空，说明遍历结束
        if (B == nullptr) return true;
        // 如果 A 不对，则 false
        if (A == nullptr || A->val != B->val) return false;
        // 继续遍历左左、右右
        return dfs(A->left, B->left) && dfs(A->right, B->right);
    }

    bool isSubStructure(TreeNode* A, TreeNode* B) {
        // 特殊 case
        if (A == nullptr || B == nullptr) return false;
        // 三种情况
        // A 包含 B 的子结构，说明 A 和 B 相等
        // A 的左子树包含 B 的子结构
        // A 的右子树包含 B 的子结构
        return dfs(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }
};
```


## [75. 颜色分类](https://leetcode.cn/problems/sort-colors/description/)

- 两次快慢双指针，或者一次相向双指针

```C++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int ptr = 0;
        for (int i = 0; i < n; ++i)
        {
            if (nums[i] == 0)
            {
                swap(nums[i], nums[ptr]);
                ++ptr;
            }
        }

        for (int i = ptr; i < n; ++i)
        {
            if (nums[i] == 1)
            {
                swap(nums[i], nums[ptr]);
                ++ptr;
            }
        }
    }
};
```


## [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/description/)

- 类似三数之和

```C++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());

        int first = 0;
        int n = nums.size();
        int diff = INT_MAX;
        int ans;
        for (; first < n; ++first)
        {
            if (first > 0 && nums[first] == nums[first - 1]) continue;

            int left = first + 1;
            int right = n - 1;
            while (left < right)
            {
                int curSum = nums[first] + nums[left] + nums[right];
                int curDiff = abs(curSum - target);
                if (curDiff < diff)
                {
                    diff = curDiff;
                    ans = curSum;
                }

                if (curSum > target) --right;
                else if (curSum < target) ++left;
                else
                {
                    return ans;
                }
            }
        }

        return ans;
    }
};
```


## [圆环回原点问题](https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg)

```
圆环上有 10 个点，编号为 0~9。从 0 点出发，每次可以逆时针和顺时针走一步，问走 n 步回到 0 点共有多少种走法。
```

```
输入: 2
输出: 2
解释：有 2 种方案。分别是 0->1->0 和 0->9->0
```

- 动态规划

```
类似于爬楼梯，回到 0 点，则第 n-1 步必须在 1 或 9
dp[i][j] 定义为 走了 i 步停在 j 的方案数
dp[i][j] = dp[i-1][(j-1+length) % length] + dp[i-1][(j+1) % length]
```

```Python
class Solution:
    def backToOrigin(self,n):
        # 点的个数为10
        length = 10
        dp = [[0 for i in range(length)] for j in range(n+1)]
        dp[0][0] = 1
        for i in range(1,n+1):
            for j in range(length):
                #dp[i][j]表示从0出发，走i步到j的方案数
                dp[i][j] = dp[i-1][(j-1+length)%length] + dp[i-1][(j+1)%length]
        return dp[n][0]
```


## [230. 二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/)

- 中序遍历是单调递增的

```C++
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> st;
        while (root != nullptr || !st.empty())
        {
            while (root)
            {
                st.push(root);
                root = root->left;
            }
            root = st.top(); st.pop();
            --k;
            if (k == 0) break;
            root = root->right;
        }
        return root->val;
    }
};
```



## [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/description/)

- 贪心，在当前位置可以调到最远的地方标记为 end，到达 end，更新

```C++
class Solution {
public:
    int jump(vector<int>& nums) {
        int maxPos = 0, n = nums.size();
        int end = 0, step = 0;
        
        // nums[0] 中的数字决定了第一步可以跳的最远的地方，记作 maxPos
        // 在行进到 maxPos 过程中，maxPos 会变化，需要 end 标记一开始的 maxPos
        for (int i = 0; i < n - 1; ++i)
        {
            maxPos = max(maxPos, nums[i] + i);
            
            if (end == i)
            {
                end = maxPos;
                ++step;
            }
        }
        
        return step;
    }
};
```


## [91. 解码方法](https://leetcode.cn/problems/decode-ways/description/)

- 动态规划，求解方案数，一般使用 dp

```C++
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        // 前 n 个解码方式
        vector<int> dp(n+1);
        
        // dp[1] += dp[0];
        dp[0] = 1;

        for (int i = 1; i <= n; ++i)
        {
            if (s[i-1] != '0')
            {
                dp[i] += dp[i-1];
            }
            if (i > 1 && s[i-2] != '0')
            {
                int num = stoi(s.substr(i-2, 2));
                if (num <= 26)
                {
                    dp[i] += dp[i-2];
                }
            }
        }

        return dp[n];
    }
};
```


## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/)

- 线性表存储

```C++
class Solution {
public:
    void preorder(TreeNode* root, vector<TreeNode*>& vec)
    {
        if (root)
        {
            vec.push_back(root);
            preorder(root->left, vec);
            preorder(root->right, vec);
        }
    }

    void flatten(TreeNode* root) {
        vector<TreeNode*> vec;
        preorder(root, vec);
        int n = vec.size();
        for (int i = 0; i < n-1; i++) {
            TreeNode *prev = vec[i], *curr = vec[i+1];
            prev->left = nullptr;
            prev->right = curr;
        }
    }
};
```

- 不需要额外空间

```C++
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* cur = root;
        while (cur != nullptr)
        {
            // 根左右，如果左子树存在
            // 左子树遍历完后，遍历右子树
            // 左子树最后一个节点就是左子树中最右边的节点
            // 将右子树拼接在该最后一个节点
            // 再把左子树移动到根节点的右边，继续遍历
            if (cur->left != nullptr)
            {
                auto nxt = cur->left;
                auto last = nxt;
                while (last->right)
                {
                    last = last->right;
                }
                last->right = cur->right;
                cur->left = nullptr;
                cur->right = nxt;
            }

            cur = cur->right;
        }
    }
};
```


## [384. 打乱数组](https://leetcode.cn/problems/shuffle-an-array/description/)

- 全排列是 n!
- 对于下标为 0 位置，从 [0,n−1] 随机一个位置进行交换，共有 n 种选择
- 下标为 1 的位置，从 [1,n−1] 随机一个位置进行交换，共有 n−1 种选择 ...

```C++
class Solution {
private:
    vector<int> raw_nums;
    vector<int> res;

public:
    Solution(vector<int>& nums) : raw_nums(nums), res(nums) {}
    
    vector<int> reset() {
        return raw_nums;
    }
    
    vector<int> shuffle() {
        // 全排列是 n!
        // 对于下标为 0 位置，从 [0,n−1] 随机一个位置进行交换，共有 n 种选择；
        // 下标为 1 的位置，从 [1,n−1] 随机一个位置进行交换，共有 n−1 种选择 ...
        for (int i = 0; i < res.size(); ++i)
        {
            int j = i + random() % (res.size() - i);
            swap(res[i], res[j]);
        }

        return res;
    }
};
```


## [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/description/)

- 26 个字母，不断创建下去  26 * 26 * 26 ……

![](https://file.fbichao.top/2024/04/9b56192494770124d6e259ccca111de4.png)

```C++
class Trie {
private:
    Trie* son[26];  // 维护 abcd……z
    bool isWord;    // 是否是一个完整单词

public:
    Trie() {
        isWord = false;                 // 初始化为空，不是单词
        for (int i = 0; i < 26; ++i)
        {
            son[i] = nullptr;
        }
    }

    ~Trie()         // 析构，释放内存
    {
        for (int i = 0; i < 26; ++i)
        {
            if (son[i] != nullptr) delete son[i];
        }
    }
    
    void insert(string word) {
        Trie* root = this;  // 一开始的 26 个
        for (char x: word)
        {
            int cur = x - 'a';
            if (root->son[cur] == nullptr) root->son[cur] = new Trie(); // 新的 26
            root = root->son[cur];  // 指向新的 26
        }
        root->isWord = true;    // 标记为一个单词
    }
    
    bool search(string word) {
        Trie* root = this;
        for (char x: word)
        {
            int cur = x - 'a';
            if (root->son[cur] == nullptr) return false;
            root = root->son[cur];
        }
        return root->isWord;    // 是单词
    }
    
    bool startsWith(string prefix) {
        Trie* root = this;
        for (char x: prefix)
        {
            int cur = x - 'a';
            if (root->son[cur] == nullptr) return false;
            root = root->son[cur];
        }
        return true;
    }
};
```



## [445. 两数相加 II](https://leetcode.cn/problems/add-two-numbers-ii/description/)

- 这里是按照数字高位到低位存储，先全部入栈，再从最低位开始计算

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 将两个链表的值存入栈中
        stack<int> st1, st2;
        ListNode* cur1 = l1;
        ListNode* cur2 = l2;
        while (cur1)
        {
            st1.push(cur1->val);
            cur1 = cur1->next;
        }
        while (cur2)
        {
            st2.push(cur2->val);
            cur2 = cur2->next;
        }

        int carry = 0;
        ListNode* newHead = nullptr;
        while (!st1.empty() || !st2.empty() || carry)
        {
            int n1 = 0, n2 = 0;
            if (!st1.empty())
            {
                n1 = st1.top();
                st1.pop();
            }

            if (!st2.empty())
            {
                n2 = st2.top();
                st2.pop();
            }
            int num = n1 + n2 + carry;
            ListNode* temp = new ListNode(num % 10);
            temp->next = newHead;
            carry = num / 10;
            newHead = temp;
        }
        return newHead;
    }
};
```


## [328. 奇偶链表](https://leetcode.cn/problems/odd-even-linked-list/description/)

- 链表的拼接，确定两个头指针，让另一个指针去指向待插入的元素

```C++
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;
        if (head->next->next == nullptr) return head;

        ListNode* odd = head;   // 奇数
        ListNode* even = head->next;    // 偶数

        ListNode* evenHead = even;  // 
        int isodd = 1;

        ListNode* cur = even->next;
        while (cur)
        {
            if (isodd)
            {
                odd->next = cur;
                odd = cur;
            }
            else
            {
                even->next = cur;
                even = cur;
            }
            isodd = !isodd;
            cur = cur->next;
        }

        odd->next = evenHead;
        even->next = nullptr;
        return head;
    }
};
```


## [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/description/)

- 动态规划，环形，偷第一家和不偷第一家

```C++
class Solution {
public:
    // 考虑两种 case
    // 包含和不包含第一家
    int robbyRange(vector<int>& nums, int left, int right)
    {
        if (left == right) return nums[left];

        vector<int> dp(nums.size(), 0);
        dp[left] = nums[left];
        dp[left+1] = max(nums[left], nums[left+1]);

        for (int i = left + 2; i <= right; ++i)
        {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }

        return dp[right];
    }

    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        
        // 不偷第一家，偷第一家
        return max(robbyRange(nums, 1, n - 1), robbyRange(nums, 0, n-2));
    }
};
```


## [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/description/)

- 原地哈希，修改原数组

```C++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();

        for (int i = 0; i < n; ++i)
        {
            while (nums[i] != (i+1))
            {
                if (nums[i] == nums[nums[i] - 1])
                {
                    return nums[i];
                }
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        return nums[n-1];
    }
};
```

====

- 快慢双指针

```C++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        while (1)
        {
            slow = nums[slow];  // 每次走一步
            fast = nums[nums[fast]]; // 每次走两步
            if (slow == fast) break;
        }

        slow = 0;
        while (slow != fast)
        {
            slow = nums[slow];
            fast = nums[fast];
        }
        return fast;
    }
};
```


## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/)

- TOP K 问题

- 小根堆

```C++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 统计频率
        unordered_map<int, int> umap;
        for (auto n: nums) umap[n]++;

        // 需要剩下 k 个大的，使用 小根堆
        auto cmp = [](pair<int,int>&m, pair<int,int>&n) {return m.second > n.second;};
        priority_queue<pair<int, int>, vector<pair<int,int>>, decltype(cmp)> q;

        // 全部入优先队列，保持个数为 k 个
        for (auto [num, count]: umap)
        {
            q.push({num, count});
            if (q.size() > k) q.pop();
        }

        // 存储结果
        vector<int> res;
        while (!q.empty())
        {
            res.push_back(q.top().first);
            q.pop();
        }

        return res;
    }
};
```

====

- 快速选择

```C++
class Solution {
public:
    void qsort(vector<pair<int,int>>& v, int start, int end, vector<int>& res, int k)
    {
        int picked = random() % (end - start + 1) + start;
        swap(v[picked], v[start]);

        int base = v[start].second;

        int index = start;
        for (int i = start + 1; i <= end; ++i)
        {
            if (v[i].second >= base)
            {
                swap(v[index + 1], v[i]);
                ++index;
            }
        }
        swap(v[start], v[index]);

        if (index - start > k)
        {
            qsort(v, start, index - 1, res, k);
        }
        else
        {
            for (int i = start; i <= index; ++i)
            {
                res.push_back(v[i].first);
            }
            if (index - start + 1 < k)
            {
                qsort(v, index + 1, end, res, k-(index-start+1));
            }
        }
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> occurances;
        for (auto v: nums)
        {
            occurances[v]++;
        }

        vector<pair<int, int>> values;
        for (auto kv: occurances) values.push_back(kv);

        vector<int> res;
        qsort(values, 0, values.size() - 1, res, k);
        return res;
    }
};
```
