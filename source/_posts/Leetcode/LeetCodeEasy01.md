---
title: LeetCode Easy(01)
tags:
  - LeetCode
  - Easy
author: fbichao
categories:
  - leetcode
  - Easy
excerpt: LeetCode Easy(01)
math: true
date: 2024-03-27 21:45:00
---
## [1. 两数之和](https://leetcode.cn/problems/two-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

> 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

> 你可以按任意顺序返回答案。

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

- 哈希表
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
哈希表存储遍历过得值和索引
```

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (umap.count(target - nums[i]))
            {
                return {i, umap[target - nums[i]]};
            }
            // 存储遍历过的值和索引
            umap[nums[i]] = i;
        }
        return {-1, -1};
    }
};
```

## [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

> 有效字符串需满足：

> 左括号必须用相同类型的右括号闭合。
> 左括号必须以正确的顺序闭合。
> 每个右括号都有一个对应的相同类型的左括号。

```
输入：s = "()[]{}"
输出：true
```

- 栈
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
用栈模拟实现
```

```C++
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;

        for (char ch: s)
        {
            // 反向入栈
            if (ch == '(') st.push(')');
            else if (ch == '{') st.push('}');
            else if (ch == '[') st.push(']');
            else
            {   // 空或元素不匹配
                if (st.empty() || st.top() != ch) return false;
                st.pop();
            }
        }

        // 例如 (()
        return st.empty();
    }
};
```

## [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

```
![](https://file.fbichao.top/2024/03/c59bb999f418f999eb450f0e4e816b4c.png)
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

- 链表归并排序
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
使用归并排序合并两个有序链表
```

```C++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* newHead = new ListNode(-1);
        ListNode* cur = newHead;
      
        // 当两个链表都是非空时
        while (list1 && list2)
        {
            if (list1->val > list2->val)
            {
                cur->next = list2;
                cur = cur->next;
                list2 = list2->next;
            }
            else
            {
                cur->next = list1;
                cur = cur->next;
                list1 = list1->next;
            }
        }

        // 剩余元素
        cur->next = list1 ? list1 : list2;

        return newHead->next;
    }
};
```

## [27. 移除元素](https://leetcode.cn/problems/remove-element/description/)

> 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

> 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

> 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
```

- 双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
快慢双指针移除元素
```

```C++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        // slow 指向不等于 val 元素的最后一个
        // fast 用于遍历
        int slow = -1, fast = 0;
        while (fast < nums.size())
        {
            // 找到不等于 val 的元素，调整 slow
            if (nums[fast] != val)
            {
                ++slow;
                nums[slow] = nums[fast];
            }
            ++fast;
        }
        return slow + 1;
    }
};
```

## [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/description/)

> 给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标（下标从 0 开始）。如果 needle 不是 haystack 的一部分，则返回  -1 。

```
输入：haystack = "sadbutsad", needle = "sad"
输出：0
解释："sad" 在下标 0 和 6 处匹配。
第一个匹配项的下标是 0 ，所以返回 0 。
```

### 暴力匹配

- 时间复杂度 $O(mn)$
- 空间复杂度 $O(1)$

```C++
class Solution {
public:
    int strStr(string haystack, string needle) {
        // 文本串 haystack，模式串 needle
        int n = haystack.size(), m = needle.size();
        // 遍历文本串，从 i 开始
        for (int i = 0; i + m <= n; i++) {
            bool flag = true;
            // 遍历模式串，从 j 开始
            for (int j = 0; j < m; j++) {
                if (haystack[i + j] != needle[j]) {
                    flag = false;
                    break;
                }
            }
            // 如果匹配
            if (flag) {
                return i;
            }
        }
        return -1;
    }
};
```

### KMP

- 时间复杂度 $O(m+n)$
- 空间复杂度 $O(|needle.size()|)$

```
文本串的遍历不会回退，所以是 $O(m)$，模式串会回退，看似时间复杂度会很高。但考虑匹配成功时，指针会向右移动一个位置，这一部分对应的时间复杂度为 $O(n+m)$。又因为向左移动的次数不会超过向右移动的次数，因此总时间复杂度仍然为 $O(n+m)$
```

```C++
class Solution {
public:
    vector<int> geneNext(string& needle)
    {
        vector<int> next(needle.size(), 0);

        int left = 0;
        for (int right = 1; right < needle.size(); ++right)
        {
            // 如果不相等，而且 left 不在 0 位置
            // left 可以表示为与后缀匹配的前缀有多少个
            while (left > 0 && needle[left] != needle[right])
            {
                //  left 已经不匹配了，返回 left 前一个位置
                left = next[left - 1];
            }

            if (needle[left] == needle[right])
            {
                ++left;
            }
            next[right] = left;
        }

        return next;
    }

    int strStr(string haystack, string needle) {
        // 对模式串 next 数组
        vector<int> next = geneNext(needle);

        int j = 0; // 模式串
        // 遍历文本串，从 i 开始
        for (int i = 0; i < haystack.size(); ++i)
        {
            // 如果 j 非 0，并且不匹配
            // 移动 j
            while (j > 0 && haystack[i] != needle[j])
            {
                j = next[j-1];
            }
            // 如果相等，移动模式串 j
            if (haystack[i] == needle[j])
            {
                ++j;
            }
            // 如果 j 大小和模式串大小一致，则返回 i - j + 1
            if (j == needle.size()) return i - j + 1;
        }

        return -1;
    }
};
```

## [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
dp[i] 表示达到台阶 i 的方法数
```

```C++
class Solution {
public:
    int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        vector<int> dp(n+1);
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < n + 1; ++i)
        {
            dp[i] = dp[i-1] + dp[i - 2];
        }

        return dp[n];
    }
};
```

## [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/)

> 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

- 中序遍历

### 递归

- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
private:
    vector<int> res;

public:
    void traversal(TreeNode* root)
    {
        if (root == nullptr) return;

        traversal(root->left);
        res.push_back(root->val);
        traversal(root->right);
    }

    vector<int> inorderTraversal(TreeNode* root) {
        traversal(root);
        return res;
    }
};
```

### 迭代

- 栈模拟
- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;

        TreeNode* cur = root;

        while(cur || !st.empty())
        {
            while (cur)     // 首先需要找到最左边的 node
            {
                st.push(cur);       // 加入节点，可以认为是左节点根节点的结合体
                cur = cur->left;
            }

            TreeNode* node = st.top(); st.pop();    // 取出节点，即左根节点
            res.push_back(node->val);
            cur = node->right;      // 遍历右节点
        }

        return res;
    }
};
```

## [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/description/)

> 给你一个二叉树的根节点 root ， 检查它是否轴对称。


![](https://file.fbichao.top/2024/03/a289e4aabab604854de69c428a5dc8b2.png)
```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

### 迭代

- 使用两个队列
- 需要两个指针
- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
public:
    bool check(TreeNode* u, TreeNode* v)
    {
        queue<TreeNode*> que;
        que.push(u);
        que.push(v);
        while (!que.empty())
        {
            TreeNode* node1 = que.front(); que.pop();
            TreeNode* node2 = que.front(); que.pop();
            if (node1 == nullptr && node2 == nullptr) continue;
            if (node1 == nullptr || node2 == nullptr) return false;
            if (node1->val != node2->val) return false;

            que.push(node1->left);
            que.push(node2->right);

            que.push(node1->right);
            que.push(node2->left);
        }

        return true;
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};
```

### 递归

- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
public:
    bool check(TreeNode* left, TreeNode* right)
    {
        if (left == nullptr && right == nullptr) return true;
        if (left == nullptr || right == nullptr) return false;
        if (left->val != right->val) return false;

        return check(left->left, right->right) && check(left->right, right->left);
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};
```

## [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/)

> 给定一个二叉树 root ，返回其最大深度。

> 二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。

![](https://file.fbichao.top/2024/03/e9cb59a669a780f34d7b047c5a27fb90.png)
```
输入：root = [3,9,20,null,null,15,7]
输出：3
```

### 迭代

- 层序遍历
- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> que;
        int res = 0;

        if (root == nullptr) return res;

        que.push(root);
        while (!que.empty())
        {
            int size = que.size();

            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = que.front(); que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            ++res;
        }

        return res;
    }
};
```

### 递归

- 先序遍历的递归
- 时间复杂度为 $O(n)$
- 空间复杂度为 $O(n)$

```C++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;

        int leftNum = maxDepth(root->left);
        int rightNUm = maxDepth(root->right);

        return max(leftNum, rightNUm) + 1;
    }
};
```

## [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/)

> 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 平衡二叉搜索树。

> 结果可能不唯一

![](https://file.fbichao.top/2024/03/789ce587c82a359d8a71c5b64e21192d.png)
```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
```

- 先序遍历的递归式
- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```
确定根节点如何选择，也就确定了结果
这里选择中间节点左边的节点作为根节点
即向下取整
```

```C++
class Solution {
public:
    TreeNode* dfs(vector<int>&nums, int left, int right)
    {
        if (left > right) return nullptr;

        int mid = left + (right - left) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = dfs(nums, left, mid - 1);
        root->right = dfs(nums, mid + 1, right);

        return root;
    }

    TreeNode* sortedArrayToBST(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        return dfs(nums, left, right);
    }
};
```


## [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/description/)

> 给定一个二叉树，判断它是否是 平衡二叉树，是指该树所有节点的左右子树的深度相差不超过 1。

- 先序遍历的递归式
- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```
使用先序遍历的递归式
```

```C++
class Solution {
public:
    int height(TreeNode* root)
    {
        if (root == nullptr) return 0;

        int leftNum = height(root->left);
        int rightNum = height(root->right);

        // 首先根据 abs(leftNum - rightNum) > 1 return - 1
        // 推迟 leftNum 和 rightNum 可能是 -1
        if (abs(leftNum - rightNum) > 1 || leftNum == -1 || rightNum == -1)
        {
            return -1;
        }
        return max(leftNum, rightNum) + 1;
    }

    bool isBalanced(TreeNode* root) {
        return height(root) != -1;
    }
};
```

## [111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/description/)

> 给定一个二叉树，找出其最小深度。

> 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

> 说明：叶子节点是指没有子节点的节点。


![](https://file.fbichao.top/2024/03/1f362ad918ac752f8e1b69c4e29fe555.png)
```
输入：root = [3,9,20,null,null,15,7]
输出：2
```

### 层序遍历

- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
队列存储节点和对应的深度，如果该节点左右都为空，则返回深度
```

```C++
class Solution {
public:
    int minDepth(TreeNode* root) {
        queue<pair<TreeNode*, int>> que;
        if (root == nullptr) return 0;

        que.push({root, 1});
        while (!que.empty())
        {
            int size = que.size();

            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = que.front().first;
                int depth = que.front().second;
                que.pop();

                if (node->left == nullptr && node->right == nullptr) return depth;

                if (node->left) que.push({node->left, depth+1});
                if (node->right) que.push({node->right, depth+1});
            }
        }

        return -1;
    }
};
```

### 前序遍历递归式

- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```C++
class Solution {
public:
    int dfs(TreeNode* root)
    {
        // 特殊 case
        if (root == nullptr) return 0;
        if (root->left == nullptr && root->right == nullptr) return 1;

        int minDepth = INT_MAX;
        if (root->left)
        {
            minDepth = min(minDepth, dfs(root->left));
        }
        if (root->right)
        {
            minDepth = min(minDepth, dfs(root->right));
        }
        return minDepth + 1;
    }

    int minDepth(TreeNode* root) {
        return dfs(root);
    }
};
```

## [112. 路径总和](https://leetcode.cn/problems/path-sum/description/)

> 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。


![](https://file.fbichao.top/2024/03/6a238d42031a13373fae5d9914984900.png)
```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
解释：等于目标和的根节点到叶节点路径如上图所示。
```

### 前序遍历迭代法
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```C++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return false;
        // 此时栈里要放的是pair<节点指针，路径数值>
        stack<pair<TreeNode*, int>> st;
        st.push({root, root->val});
        while (!st.empty()) {
            auto node = st.top(); st.pop();
            
            // 如果该节点是叶子节点了，同时该节点的路径数值等于targetSum，那么就返回true
            if (node.first->left == nullptr && node.first->right == nullptr)
                if (targetSum == node.second) return true;

            // 右节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if (node.first->right) {
                st.push(pair<TreeNode*, int>(node.first->right, node.second + node.first->right->val));
            }

            // 左节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if (node.first->left) {
                st.push(pair<TreeNode*, int>(node.first->left, node.second + node.first->left->val));
            }
        }
        return false;
    }
};
```

### 前序遍历递归法
- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```C++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return false;
        if (root->left == nullptr && root->right == nullptr) return targetSum == root->val;

        bool leftPath = hasPathSum(root->left, targetSum - root->val);
        bool rightPath = hasPathSum(root->right, targetSum - root->val);
        return leftPath || rightPath;
    }
};
```



## [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

> 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

> 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
每天有两种状态，不持有或持有
```

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2, 0));

        // 不持有、持有
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < n; ++i)
        {
            // 和前一天比较
            // 前一天持有，今天卖，前一天不持有，今天保持
            dp[i][0] = max(prices[i]+dp[i-1][1], dp[i-1][0]);
            // 前一天持有，今天保持，前一天不持有，今天持有
            dp[i][1] = max(dp[i-1][1], -prices[i]);
        }

        return dp[n-1][0];
    }
};
```

## [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

> 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

```
输入：nums = [4,1,2,1,2]
输出：4
```

- 异或操作
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
异或 `^` 操作
0 ^ a = a;
a ^ a = 0;
```

```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (auto num: nums) res ^= num;
        return res;
    }
};
```

## [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个链表的头节点 head ，判断链表中是否有环。

> 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

> 如果链表中存在环 ，则返回 true 。 否则，返回 false 。

![](https://file.fbichao.top/2024/03/762ab93fb1f80317cecd225f2c7d4809.png)
```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

- 快慢双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
快慢双指针，快指针每次多走一步
```

```C++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if (head == nullptr) return false;
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }
};
```
