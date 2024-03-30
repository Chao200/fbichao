---
title: LeetCode Mid(3)
tags:
  - LeetCode
  - Mid
author: fbichao
categories:
  - leetcode
  - Mid
excerpt: LeetCode Mid(3)
math: true
date: 2024-03-13 21:45:00
---
## [72. 编辑距离](https://leetcode.cn/problems/edit-distance/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数。

> 你可以对一个单词进行如下三种操作：

> 插入一个字符
> 删除一个字符
> 替换一个字符

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

- 动规
- 时间复杂度 $O(mn)$
- 空间复杂度 $O(mn)$

```
1. 按照两个字符串结尾划分 dp[i][j]，即 word1 的前 i 个构成的 str1 变成 word2 的前 j 个 str2 所需的最少操作数

2. 状态转移
看 word1[i-1] 和 word2[j-1] 是否相等
case1. 相等 dp[i][j] = dp[i-1][j-1]
case2. 不相等，执行替换操作 dp[i][j] = dp[i-1][j-1] + 1
case3. 不相等，执行插入操作 dp[i][j] = dp[i-1][j] + 1
case4. 不相等，执行删除操作 dp[i][j] = dp[i][j-1] + 1

3. 初始化
空字符串，执行插入操作
- 当 i = 0，dp[0][j] = j; 
- 当 j = 0，dp[i][0] = i;
```

```C++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();

        // dp[i][j]
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

        // 初始化
        for (int i = 0; i <= m; ++i) dp[i][0] = i;
        for (int i = 0; i <= n; ++i) dp[0][i] = i;

        // 遍历
        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            { 
                // 状态转移
                if (word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1];
                else dp[i][j] = min({dp[i-1][j-1] + 1, dp[i-1][j]+1, dp[i][j-1]+1});
            }
        }

        return dp[m][n];
    }
};
```

## [75. 颜色分类](https://leetcode.cn/problems/sort-colors/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

> 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

> 必须在不使用库内置的 sort 函数的情况下解决这个问题。

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

- 双指针 + 快排思想
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
快速排序选择 base 后，将小于 base 置于左侧，大于 base 置于右侧
swap 的时候，只有 left 和 right 是确定的，可以移动，index 位置的值是交换后的信值，不确定，不能 index++
所以可能会有 index < left 出现，需要 ++index
```

```C++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        int index = 0;
        while (index <= right)
        {
            // if
            if (index < left)
            {
                ++index;
            }
            else if (nums[index] == 0)
            {
                swap(nums[index], nums[left]);
                ++left;
            }
            else if (nums[index] == 2)
            {
                swap(nums[index], nums[right]);
                --right;
            }
            else
            {
                ++index;
            }
        }
    }
};
```

## [77. 组合](https://leetcode.cn/problems/combinations/description/)

> 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

```
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

```C++
class Solution {
private:
    // 存储结果
    vector<vector<int>> res;
    // 存储路径
    vector<int> path;
    // 来源
    vector<int> nums;
    
public:
    void backtrack(int n, int k, int startIndex)
    {   // 满足要求
        if (path.size() == k)
        {
            res.push_back(path);
            return;
        }

        for (int i = startIndex; i < n; ++i)
        {
            path.push_back(nums[i]);
            backtrack(n, k, i+1);
            path.pop_back();
        }
    }

    vector<vector<int>> combine(int n, int k) {
        for (int i = 1; i <= n; ++i) nums.push_back(i);
        backtrack(n, k, 0);
        return res;
    }
};
```



## [78. 子集](https://leetcode.cn/problems/subsets/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

> 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

- 回溯
- 时间复杂度 $O(n\cdot{2^n})$
- 空间复杂度 $O(n)$

```
遍历从当前 index 开始，递归一样
```

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    // 递归从 index 元素开始
    void backtrack(vector<int>& nums, int index)
    {
        res.push_back(path);
    
        // 遍历从当前元素开始往后
        for (int i = index; i < nums.size(); ++i)
        {
            path.push_back(nums[i]);
            backtrack(nums, i+1);
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        backtrack(nums, 0);
        return res;
    }
};
```

## [79. 单词搜索](https://leetcode.cn/problems/word-search/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

> 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```
![](https://file.fbichao.top/2024/03/729db9916ff1a64a5a6d1ad4e8315e08.png)
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

- 回溯
- 时间复杂度 $O(3^l\cdot{mm})$，$l$ 是 word 长度，因为除了第一次 dfs 是四个方向，其余都是三个方向，因为回头的方向直接 return 了，所以是三个方向，每个方向又会分三个方向，总共长 $l$
- 空间复杂度 $O(mn)$

```
注意 dfs 中 board 设置的值
先确定 row 和 col 才可以访问 board 数组
```

```C++
class Solution {
public:
    bool dfs(vector<vector<char>>& board, string word, int row, int col, int index)
    {
        if (index == word.size()) return true;
        // 一定要先确定 row 和 col 合法
        if (row < 0 || row >= board.size()) return false;
        if (col < 0 || col >= board[0].size()) return false;
        if (board[row][col] != word[index]) return false;

        // 设置为空字符，这样其余 dfs 就不会产生回头
        board[row][col] = '\0';

        bool res = dfs(board, word, row+1, col, index + 1)
                    || dfs(board, word, row-1, col, index + 1)
                    || dfs(board, word, row, col+1, index + 1)
                    || dfs(board, word, row, col-1, index + 1);
    
        // 还原
        board[row][col] = word[index];

        return res;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // 从某个字符开始如果可以构成 word，则 return true
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
};
```

## [90. 子集 II](https://leetcode.cn/problems/subsets-ii/description/)

> 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的 子集（幂集）。

> 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

```
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

```
不重复，就排序+used
```


```C++
class Solution {
private:
    vector<vector<int>>  res;
    vector<int> path;

public:
    void backtrack(vector<int>& nums, vector<bool>& used, int index)
    {
        res.push_back(path);
        
        if (index == nums.size()) return;

        for (int i = index; i < nums.size(); ++i)
        {
            if (i > 0 && nums[i] == nums[i-1] && used[i-1] == false) continue;
            used[i] = true;
            path.push_back(nums[i]);
            backtrack(nums, used, i + 1);
            path.pop_back();
            used[i] = false;
        }
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        sort(nums.begin(), nums.end());
        backtrack(nums, used, 0);

        return res;
    }
};
```




## [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/description/)

> 有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

> 例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
> 给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。


```
输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
```


```C++
class Solution {
private:
    vector<string> res;
    vector<string> path;

public:
    // 判断字符串是否合法
    bool isValid(string& s, int start, int end)
    {
        if (start > end) return false;

        if (s[start] == '0' && start != end) return false;

        int num = 0;
        for (int i = start; i <= end; ++i)
        {
            if (s[i] > '9' || s[i] < '0') return false;
            num = num * 10 + s[i] - '0';
            if (num > 255) return false;
        }

        return true;
    }

    void backtrack(string& s, int index)
    {
        if (path.size() > 4) return;
        if (path.size() == 4 && index == s.size())
        {
            string temp = path[0];
            for (int i = 1; i < 4; ++i)
            {
                temp += ".";
                temp += path[i];
            }
            res.push_back(temp);
            return;
        }

        for (int i = index; i < s.size(); ++i)
        {
            if (isValid(s, index, i))
            {
                path.push_back(s.substr(index, i - index + 1));
                backtrack(s, i + 1);
                path.pop_back();
            }
        }
    }

    vector<string> restoreIpAddresses(string s) {
        if (s.size() < 4 || s.size() > 12) return res;
        backtrack(s, 0);
        return res;
    }
};
```






## [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

```
![](https://file.fbichao.top/2024/03/bd30d760a25bb5595faf4b661f1a6b0c.png)
输入：n = 3
输出：5
```

- 动规
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n)$

```
1. 定义 dp[i] 表示整数 i 构成的二叉搜索树个数
2. 状态转移 dp[i] += dp[j] * dp[j - i -1]; j 从 0~i-1
比如 i = 3，分解为 
========= i ========= j =========== j-i-1===
1）根节点为 1，左子树节点 0 个，右子树节点 2 个
2）根节点为 2，左子树节点 1 个，右子树节点 1 个
3）根节点为 3，左子树节点 2 个，右子树节点 0 个
```

```C++
class Solution {
public:
    int numTrees(int n) {
        // dp[3] = d[0] * d[2] + d[1] * dp[1] + dp[2] * dp[0]

        vector<int> dp(n+1);
        dp[0] = 1;
        for (int i = 1; i <= n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                // dp[i] = sum(j) dp[j] * dp[i-j-1]
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }
};
```

## [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

> 有效 二叉搜索树定义如下：

> 节点的左子树只包含 小于 当前节点的数。节点的右子树只包含 大于 当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。

```
![](https://file.fbichao.top/2024/03/ec70849eb20e54df14dbe3ff9e027ab2.png)
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
```

### 递归

- 递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
前序遍历的递归，每个节点都有一个区间，不断更新这个区间
注意使用 long 类型
```

```C++
class Solution {
public:
    bool dfs(TreeNode* root, long left, long right)
    {
        if (root == nullptr) return true;
        if (root->val >= right || root->val <= left) return false;

        return dfs(root->left, left, root->val) &&
                dfs(root->right, root->val, right);
    }

    bool isValidBST(TreeNode* root) {
        return dfs(root, LONG_MIN, LONG_MAX);
    }
};
```

### 迭代

- 中序遍历二叉搜索树，得到的是升序数组
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
中序遍历二叉搜索树，得到的是升序数组
```

```C++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* cur = root;
        // long 类型
        long prev = LONG_MIN;
    
        // 中序遍历的迭代形式
        while (!st.empty() || cur)
        {
            while (cur)
            {
                st.push(cur);
                cur = cur->left;
            }

            cur = st.top(); st.pop();

            if (prev >= cur->val) return false;

            // 更新
            prev = cur->val;
            cur = cur->right;
        }

        return true;
    }
};
```

## [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

```
![](https://file.fbichao.top/2024/03/a5cbff934cb602159839b4b4b06b1437.png)
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

- 层次遍历
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
对每层遍历，所以 while 里面还嵌套循环
```

```C++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if (root == nullptr) return {};

        // 队列保存节点
        queue<TreeNode*> que;
        que.push(root);
    
        vector<vector<int>> res;
        while (!que.empty())
        {
            int n = que.size();

            vector<int> temp;
            // 对每层遍历
            for (int i = 0; i < n; ++i)
            {
                auto node = que.front(); que.pop();
                temp.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            res.push_back(temp);
        }

        return res;
    }
};
```

## [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

```
![](https://file.fbichao.top/2024/03/bf666fd69dd13bd926de7cbaa7a8d578.png)
输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]
```

- 前序遍历的递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
先找根节点，再找根节点两侧的节点
使用左闭右开区间
```

```C++
class Solution {
private:
    unordered_map<int, int> umap;

public:
    TreeNode* build(vector<int>& preorder, vector<int>& inorder, 
                int pre_l, int pre_r, int ind_l, int ind_r)
    {
        if (ind_l == ind_r) return nullptr;
        int size = umap[preorder[pre_l]] - ind_l;
    
        TreeNode* leftTree = build(preorder, inorder,
                            pre_l + 1, pre_l + 1 + size,
                            ind_l, ind_l + size);
        TreeNode* rightTree = build(preorder, inorder,
                            pre_l + 1 + size, pre_r,
                            ind_l + size + 1, ind_r);

        return new TreeNode(preorder[pre_l], leftTree, rightTree);
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = inorder.size();
        for (int i = 0; i < n; ++i) umap[inorder[i]] = i;

        // 左闭右开
        return build(preorder, inorder, 0, n, 0, n);
    }
};
```

## [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)

> 给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。

- 先序遍历递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
类似 [105-从前序与中序遍历序列构造二叉树](#105-从前序与中序遍历序列构造二叉树)
```

```C++
class Solution {
private:
    unordered_map<int, int> index;

public:
    TreeNode* dfs(vector<int> &inorder, vector<int> &postorder, 
                        int in_l, int in_r, int post_l, int post_r)
    {
        if (in_l == in_r) return nullptr;

        int left_size = index[postorder[post_r - 1]] - in_l; // 左子树的大小
        TreeNode *left = dfs(inorder, postorder, 
                            in_l, in_l + left_size, 
                            post_l, post_l + left_size
                            );
        TreeNode *right = dfs(inorder, postorder, 
                            in_l + left_size + 1, in_r, 
                            post_l + left_size, post_r - 1
                            );

        return new TreeNode(postorder[post_r - 1], left, right);
    };

    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
        int n = inorder.size();

        for (int i = 0; i < n; i++) {
            index[inorder[i]] = i;
        }

        return dfs(inorder, postorder, 0, n, 0, n); // 左闭右闭区间
    }
};

```








## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你二叉树的根结点 root ，请你将它展开为一个单链表：

> 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
> 展开后的单链表应该与二叉树 先序遍历 顺序相同。

```
![](https://file.fbichao.top/2024/03/1ce11da4fdf8be8abb17ce07f2845685.png)
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

- 先序遍历递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
先序遍历的递归，先得到节点的先序遍历
再拼接
```

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
        for (int i = 1; i < n; i++) {
            TreeNode *prev = vec[i-1], *curr = vec[i];
            prev->left = nullptr;
            prev->right = curr;
        }
    }
};
```

## [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/)

> 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

> 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

> 返回 你能获得的 最大 利润 。

```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。
```

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // 每天要么是不持有、要么持有
        vector<vector<int>> dp(n, vector<int>(2, 0));

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
        }

        return dp[n-1][0];
    }
};
```




## [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

> 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

- 原地哈希
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
使用 unordered_set 记录集合
遍历集合，如果当前比元素小 1 的元素可以在集合中找到，就继续遍历
如果找不到说明，当前元素是该序列的起点，则进入循环求解长度和值
```

```C++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> uset;
        for (int num: nums) uset.insert(num);

        int res = 0;

        // 遍历集合
        for (auto num: uset)
        {
            // 如果不是序列的起点
            if (!uset.contains(num - 1))
            {
                int currNum = num;
                int currLen = 1;

                // 循环求解长度
                while (uset.contains(currNum + 1))
                {
                    currNum++;
                    currLen++;
                }
            
                res = max(res, currLen);
            }
        }

        return res;
    }
};
```

## [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/description/)


> 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

```
可以允许重复
for 循环遍历的是同一层，所以需要 substr
```


```C++
class Solution {
private:
    vector<vector<string>> res;
    vector<string> path;

public:
    bool is_hui(string s)
    {
        if (s.size() == 1) return true;
        int left = 0, right = s.size() - 1;
        while (left < right)
        {
            if (s[left++] != s[right--]) return false;
        }

        return true;
    }

    void backtrack(string s, int index)
    {
        if (index == s.size())
        {
            res.push_back(path);
            return;
        }

        for (int i = index; i < s.size(); ++i)
        {
            // 从节点 index 开始的每个子串
            string str = s.substr(index, i - index + 1); 
            if (is_hui(str))
            {
                path.push_back(str);
            }
            else continue;
            backtrack(s, i+1);
            path.pop_back();
        }
    }

    vector<vector<string>> partition(string s) {
        backtrack(s, 0);
        return res;
    }
};
```




