---
title: LeetCode Easy(03)
tags:
  - LeetCode
  - Easy
author: fbichao
categories:
  - leetcode
  - Easy
excerpt: LeetCode Easy(03)
math: true
date: 2024-03-29 21:45:00
---
## [617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你两棵二叉树： root1 和 root2 。

> 想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。

> 返回合并后的二叉树。

> 注意: 合并过程必须从两个树的根节点开始。

![](https://file.fbichao.top/2024/03/4842fdc8558658701bf7e692d1835370.png)
```
输入：root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
输出：[3,4,5,5,4,null,7]
```

- 先序遍历递归
- 时间复杂度 $O(min(m, n))$
- 空间复杂度 $O(min(h_m, h_n))$

```
先序遍历递归式
```

```C++
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if (root1 == nullptr) return root2;
        if (root2 == nullptr) return root1;

        auto merge = new TreeNode(root1->val + root2->val);
        merge->left = mergeTrees(root1->left, root2->left);
        merge->right = mergeTrees(root1->right, root2->right);
        return merge;
    }
};
```

## [674. 最长连续递增序列](https://leetcode.cn/problems/longest-continuous-increasing-subsequence/description/)

> 给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

> 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

```
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。
```


```C++
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);
        dp[0] = 1;
        int res = 1;

        for (int i = 1; i < n; ++i)
        {
            if (nums[i] > nums[i-1])
                dp[i] = dp[i-1] + 1;
            res = res > dp[i] ? res : dp[i];
        }

        return res;
    }
};
```




## [700. 二叉搜索树中的搜索](https://leetcode.cn/problems/search-in-a-binary-search-tree/description/)

> 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。

> 你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。

```C++
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        TreeNode* cur = root;
        while (cur)
        {
            if (cur->val > val) cur = cur->left;
            else if (cur->val < val) cur = cur->right;
            else break;
        }
        return cur;
    }
};
```


## [704. 二分查找](https://leetcode.cn/problems/binary-search/description/)

> 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。


```
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
```

- 二分查找
- 时间复杂度 $O(logn)$
- 空间复杂度 $O(1)$

```
有序 + 二分
```

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) right = mid - 1;
            else if (nums[mid] < target) left = mid + 1;
            else return mid;
        }

        return -1;
    }
};
```

## [746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/description/)

> 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

> 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。

> 请你计算并返回达到楼梯顶部的最低花费。

```
输入：cost = [1,100,1,1,1,100,1,1,100,1]
输出：6
解释：你将从下标为 0 的台阶开始。
- 支付 1 ，向上爬两个台阶，到达下标为 2 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 4 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 6 的台阶。
- 支付 1 ，向上爬一个台阶，到达下标为 7 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 9 的台阶。
- 支付 1 ，向上爬一个台阶，到达楼梯顶部。
总花费为 6 。
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```

```

```C++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n+1);
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; ++i)
        {
            dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]);
        }
        return dp[n];
    }
};
```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## [977. 有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/description/)

> 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

```
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]
```

- 双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
通过双指针将新的结果存入
```

```C++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> res(nums.size(), 0);
        int size = nums.size() - 1;
        int left = 0, right = size;
        while (left <= right)
        {
            if (nums[left] * nums[left] > nums[right] * nums[right])
            {
                res[size--] = nums[left] * nums[left];
                ++left;
            }
            else if (nums[left] * nums[left] <= nums[right] * nums[right])
            {
                res[size--] = nums[right] * nums[right];
                --right;
            }
        }
        return res;
    }
};
```




## [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/description/)

> 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

> 在 S 上反复执行重复项删除操作，直到无法继续删除。

> 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。 

```
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"
```

- 使用栈模拟
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
使用栈模拟，如果入栈元素和栈顶元素相同，则 pop
最后组装栈中剩余元素
```

```C++
class Solution {
public:
    string removeDuplicates(string s) {
        // 删除
        stack<char> st;
        for (char c: s)
        {
            if (!st.empty() && st.top() == c) st.pop();
            else st.push(c);
        }

        // 组装
        string res;
        while (!st.empty())
        {
            res = st.top() + res;
            st.pop();
        }
        return res;
    }
};
```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```




## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```



## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```




## []()

- 

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```
