---
title: LeetCode Mid(5)
tags:
  - LeetCode
  - Mid
author: fbichao
categories:
  - leetcode
  - Mid
excerpt: LeetCode Mid(5)
math: true
date: 2024-03-19 21:45:00
---
## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

> 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

> 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

```
输入: [3,2,1,5,6,4], k = 2
输出: 5
```

### 优先队列

- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
使用优先队列维护大小为 k 的小根堆，这样最后剩下的头元素就是第 k 大
```

```C++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // 小根堆
        auto cmp = [](int a, int b)
        {
            return a > b;
        };
        priority_queue<int, vector<int>, decltype(cmp)> prio_que;

        for (int i = 0; i < nums.size(); ++i)
        {
            // 不足 k 个元素直接 push
            if (prio_que.size() < k)
            {
                prio_que.push(nums[i]);
            }
            // 足够 k 个，如果当前元素大于顶元素
            else if (nums[i] > prio_que.top())
            {
                prio_que.pop();
                prio_que.push(nums[i]);
            }
        }

        return prio_que.top();
    }
};
```

### 快排

- 时间复杂度 $O(n)$
- 空间复杂度 $O(logn)$

```
期望为线性的选择算法，和快排还不一样
当划分的点右侧只有 k-1 个元素，就找到了
```

```C++
class Solution {
public:
    int quickselect(vector<int> &nums, int l, int r, int k) {
        if (l == r) return nums[l];
        int partition = nums[l], i = l - 1, j = r + 1;
        while (i < j) {
            do i++; while (nums[i] < partition);
            do j--; while (nums[j] > partition);
            if (i < j)
                swap(nums[i], nums[j]);
        }
        // 如果 k<= j 在左半边
        if (k <= j)return quickselect(nums, l, j, k);
        // 在右半边
        else return quickselect(nums, j + 1, r, k);
    }

    int findKthLargest(vector<int> &nums, int k) {
        int n = nums.size();
        // 第 k 大，就是从小到大第 n-k个
        return quickselect(nums, 0, n - 1, n - k);
    }
};
```

## [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/description/)

> 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：

> 只使用数字 1 到 9 每个数字 最多使用一次，返回 所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。



```
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
解释:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
没有其他符合的组合了。
```

```
数据源是 1~9，不重复
backtrack 每次需要后移
```


```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    int sum(vector<int>& tmp)
    {   // 求和
        int sn = 0;
        for (auto n: tmp)
            sn += n;
        return sn;
    }

    void backtrack(int k, int n, int index)
    {
        if (path.size() == k)
        {
            if (sum(path) == n)
                res.push_back(path);
            return;
        }

        // 剩余数字和加入 path 的数字个数必须不小于 k 个
        for (int i = index; 9 - i + 1 + path.size() >= k; ++i)
        {
            path.push_back(i);
            backtrack(k, n, i + 1);
            path.pop_back();
        }
    }

    vector<vector<int>> combinationSum3(int k, int n) {
        backtrack(k, n, 1);
        return res;
    }
};
```




## [221. 最大正方形](https://leetcode.cn/problems/maximal-square/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。

```
![](https://file.fbichao.top/2024/03/1da959fe044c178eb5a47a4bb1387526.png)
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4
```

- 动态规划
- 时间复杂度 $O(mn)$
- 空间复杂度 $O(mn)$

```
dp[i][j] 表示 i 行 j 列的正方形大小
边界上的正方形大小为 1
内部一个方格，若加上该方格可以是正方形，那么必定上左、上、左都是 1，取 min
```

```C++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        int res = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (matrix[i][j] == '1')
                {
                    // 边界
                    if (i==0||j==0) dp[i][j] = 1;
                    // 内部
                    else dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1;
                }
                res = max(res, dp[i][j]);
            }
        }
        return res * res;
    }
};
```

## [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)

> 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

```
可以使用 [236-二叉树的最近公共祖先](#236-二叉树的最近公共祖先)
这里可以利用二叉树性质
```

- 二叉搜索树性质
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* res = root;
        while (1)
        {
            if (res->val > p->val && res->val > q->val) res = res->left;
            else if (res->val < p->val && res->val < q->val) res = res->right;
            else break;
        }
        return res;
    }
};
```


## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

> 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

```
![](https://file.fbichao.top/2024/03/17da25b38a25d4d85314ccf0fb043a74.png)
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

- 先序遍历递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
先序遍历的递归形式

两种 case：一种是 root 是 p 和 q 的最近祖先，一种 p 或 q 是另一个的最近祖先
base case：root == nullptr、p、q
左子树返回一个，右子树返回一个，看是否为 nullptr
```

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // ① 根节点就是 p 或 q
        // ② p 和 q 在 root 的两侧
        if (root == nullptr) return nullptr;
        if (root == p || root == q) return root;

        auto leftNode = lowestCommonAncestor(root->left, p, q);
        auto rightNode = lowestCommonAncestor(root->right, p, q);

        if (leftNode && rightNode) return root;
        return leftNode == nullptr ? rightNode : leftNode;
    }
};
```

## [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

> 题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

> 请 不要使用除法，且在 O(n) 时间复杂度内完成此题。

```
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> res(nums.size(), 1);
      
        // 计算每个数左侧的乘积
        int left = 1;
        for (int i = 0; i < nums.size(); ++i)
        {
            // left 后更新
            res[i] *= left;
            left *= nums[i];
        }
      
        // 计算每个数右侧的乘积
        int right = 1;
        for (int i = nums.size() - 1; i >= 0; --i)
        {
            res[i] *= right;
            right *= nums[i];
        }

        return res;
    }
};
```

## [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

> 每行的元素从左到右升序排列。
> 每列的元素从上到下升序排列。

```
![](https://file.fbichao.top/2024/03/70bd40d74a3b7b127166a608ac84fb21.png)
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```

- 暴力搜索($O(mn)$)、对每行二分查找($O(mlogn)$)、对每列二分查找($O(nlogm)$)
- 从右上角往左下角搜索
- 时间复杂度 $O(m+n)$
- 空间复杂度 $O(1)$

```
从右上角往左下角搜
如果当前位置大于 target，那么右侧和下侧不需要搜索，将 --y
如果当前位置小于 target，此时右侧没有数据，左侧都比自己小，往下移动 ++x
```

```C++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        int x = 0, y = cols - 1;
        while (x < rows && y >= 0)
        {
            if (matrix[x][y] == target) return true;
            if (matrix[x][y] > target) --y;
            else ++x;
        }

        return false;
    }
};
```

## [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

> 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

- 动规，两阶段
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n)$

```
拆分形式的动规
```

```C++
class Solution {
public:
    bool isSquare(int n)
    {
        if (n == 1) return true;
        for (int i = 1; i <= n / 2; ++i)
        {
            if (i * i == n) return true;
        }
        return false;
    }

    int numSquares(int n) {
        // 返回的是最少数量，所以需要取 MAX
        vector<int> dp(n+1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= n; ++i)
        {
            // 不是完全平方数
            if (!isSquare(i)) continue;
            // 如果 i 是完全平方数，那么 j 就可以拆成 i + j - i
            for (int j = i; j <= n; ++j)
            {
                // 如果 j-i 可以由平方数构成
                if (dp[j-i] != INT_MAX)
                    dp[j] = min(dp[j], dp[j-i]+1);
            }
        }

        return dp[n] == INT_MAX ? 0 : dp[n];
    }
};
```

- 时间复杂度 $O(n\sqrt{n})$

```C++
class Solution {
public:
    int numSquares(int n) {
        // 返回的是最少数量，所以需要取 MAX
        vector<int> dp(n+1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j * j <= i; ++j)
            {
                if (dp[i - j * j] != INT_MAX)
                {
                    dp[i] = min(dp[i - j * j] + 1, dp[i]);
                }
            }
        }

        return dp[n] == INT_MAX ? 0 : dp[n];
    }
};
```

## [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。

> 假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。

> 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

```
输入：nums = [3,1,3,4,2]
输出：3
```

### 二分查找

- 时间复杂度 $O(n\cdot{logn})$
- 空间复杂度 $O(1)$

```
二分查找，要查找的数字在 `1~n` 之间
如果数字 i 出现次数大于了 i 次，就说明大于等于数字 i 的次数都会超过本身
小于 i 的数字次数都会小于等于自身
```

```C++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int left = 1, right = n - 1;
        int ans = -1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            int cnt = 0;
            for (int i = 0; i < n; ++i)
            {
                if (nums[i] <= mid) ++cnt;
            }

            // 如果数字不重复，则等号成立
            // 如果数字不出现，则小于号成立
            if (cnt <= mid) left = mid + 1;
            // 出现重复数字，right 右移动，ans 可能是 mid
            else
            {
                right = mid - 1;
                ans = mid;
            }
        }
        return ans;
    }
};
```

### 双指针

- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
- 数组形式的判圈和找入口
- Floyd 判圈法
```

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

## [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

> 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

- 动态规划
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n)$

```
dp[i] 表示以 nums[i] 结尾的最长子串，固定了结尾，遍历开始地方
```

```C++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        vector<int> dp(n, 1);

        int res = 1;

        // 固定结尾
        for (int i = 1; i < n; ++i)
        {
            // 遍历开始
            for (int j = 0; j < i; ++j)
            {
                if (nums[i] > nums[j])
                {
                    dp[i] = max(dp[i], dp[j] + 1);
                    res = max(res, dp[i]);
                }
            }
        }

        return res;
    }
};
```

## [309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。

> 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

> 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```
输入: prices = [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
要么持有，要么不持有，不持有，要么冷冻期，要么非冷冻期
三种状态
```

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;

        int n = prices.size();

        vector<vector<int>> dp(n, vector<int>(3, 0));
        /*
            要么持有，要么不持有，不持有，要么冷冻期，要么非冷冻期
        */
        // dp[i][0]: 手上持有股票的最大收益
        // dp[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
        // dp[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
        dp[0][0] = -prices[0];
        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = max(dp[i-1][0], dp[i-1][2] - prices[i]);
            dp[i][1] = dp[i-1][0] + prices[i];
            dp[i][2] = max(dp[i-1][1], dp[i-1][2]);
        }

        return max(dp[n-1][1], dp[n-1][2]);
    }
};
```
