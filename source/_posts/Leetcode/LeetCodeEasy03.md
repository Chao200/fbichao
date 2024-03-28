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

```
![](https://file.fbichao.top/2024/03/4842fdc8558658701bf7e692d1835370.png)
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

## []()

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

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++


```
