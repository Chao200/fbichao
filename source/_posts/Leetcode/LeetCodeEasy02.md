---
title: LeetCode Easy(02)
tags:
  - LeetCode
  - Easy
author: fbichao
categories:
  - leetcode
  - Easy
excerpt: LeetCode Easy(02)
math: true
date: 2024-03-28 21:45:00
---

## [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

> 图示两个链表在节点 c1 开始相交：
![](https://file.fbichao.top/2024/03/e894bf9bd8faa691a3ce9afb894897f2.png)

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
```

- 分离双指针
- 时间复杂度 $O(n+m)$
- 空间复杂度 $O(1)$

```
使用分离双指针，在一个链表中结束后，指向另一个链表继续
```

```C++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* p1 = headA;
        ListNode* p2 = headB;
        // 在两个链表中遍历
        while (p1 != p2)
        {
            p1 = p1 ? p1->next : headB;
            p2 = p2 ? p2->next : headA;
        }
        return p1;
    }
};
```





## [169. 多数元素](https://leetcode.cn/problems/majority-element/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

> 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```
输入：nums = [3,2,3]
输出：3
```

- 哈希表
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
哈希表记录每个元素出现次数
```

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> umap;
        // 记录当前次数最多的元素和次数
        int res = 0, cnt = 0;
        for (int num: nums)
        {
            ++umap[num];
            if (umap[num] > cnt)
            {
                res = num;
                cnt = umap[num];
            }
        }
        return res;
    }
};
```




## [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/description/)

> 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。


```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

- 迭代和递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
迭代
```

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while (cur)
        {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }

        return prev;
    }
};
```




## [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。



```
![](https://file.fbichao.top/2024/03/1f7d5e15bf17f2c0983eafc8b08898d3.png)
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

### 递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
先序遍历的递归式
```

```C++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) return root;

        // 反转左子树
        auto left = invertTree(root->left);
        // 反转右子树
        auto right = invertTree(root->right);

        root->left = right;
        root->right = left;

        return root;
    }
};
```

### 迭代
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
栈模拟
```

```C++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) return nullptr;

        queue<TreeNode*> st;
        st.push(root);
        while (!st.empty())
        {
            TreeNode* node = st.front(); st.pop();

            if (node->left) st.push(node->left);
            if (node->right) st.push(node->right);

            // 反转当前节点左右子树
            swap(node->left, node->right);
        }

        return root;
    }
};
```


## [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

```
输入：head = [1,2,2,1]
输出：true
```

- 反转链表+双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
- 通过快慢双指针找到中间节点的前一个节点
- 反转后半个链表
- 断开成两个链表
- 分离双指针比较值
```

```C++
class Solution {
public:
    ListNode* getMiddle(ListNode* head)
    {
        // 得到中间节点的前一个节点
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast->next && fast->next->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

    ListNode* reverse_list(ListNode* head)
    {   
        // 反转链表，返回反转后的头结点
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while (cur)
        {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }
        return prev;
    }

    bool isPalindrome(ListNode* head) {
        if (head == nullptr) return true;
        ListNode* mid_prev = getMiddle(head);
        ListNode* end = reverse_list(mid_prev->next);
        mid_prev->next = nullptr;
        ListNode* cur = head;
        ListNode* reverse_head = end;
        while (reverse_head)
        {
            if (cur->val != reverse_head->val) 
            {
                return false;
            }
            cur = cur->next;
            reverse_head = reverse_head->next;
        }

        return true;
    }
};
```




## [283. 移动零](https://leetcode.cn/problems/move-zeroes/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

> 请注意 ，必须在不复制数组的情况下原地对数组进行操作。

```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

- 快慢双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
slow 指向的是非零元素的最后一个，fast 用于遍历
```

```C++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        // slow 指向的是非零元素的最后一个
        // fast 用于遍历
        int slow = -1, fast = 0;

        while (fast < nums.size())
        {
            // 如果不是 0，前移
            if (nums[fast]!=0)
            {
                ++slow;
                swap(nums[slow], nums[fast]);
            }
            ++fast;
        }
    }
};
```




## [338. 比特位计数](https://leetcode.cn/problems/counting-bits/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。

```
输入：n = 5
输出：[0,1,1,2,1,2]
解释：
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
对每个数都求模、求商的过程，计算 1 的个数，复杂度是 $O(nlogn)$
动规实现 $O(n)$
利用奇偶数的最低位是 1\0
```

```C++
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> dp(n+1, 0);
        for (int i = 1; i <= n; ++i)
        {
            // 如果 i 是偶数，则最低位是 0
            // 如果 i 是奇数，则最低位是 1
            dp[i] = dp[i >> 1] + (i & 1);
        }
    }
};
```




## [448. 找到所有数组中消失的数字](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。

```
输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6]
```

- 原地哈希
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
对数组中存在的元素，将相对应索引位置的值+n
```

```C++
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        // 对于数组中的元素，将对应索引的值+n
        // 由于可能重复数字，所以取模
        for (auto num: nums)
        {
            int index = (num - 1) % n;
            nums[index] += n;
        }

        // 对每个位置遍历，若元素小于 n，则找到了
        vector<int> res;
        for (int i = 0; i < n; ++i)
        {
            if (nums[i] <= n)
            {
                res.push_back(i+1);
            }
        }

        return res;
    }
};
```



## [461. 汉明距离](https://leetcode.cn/problems/hamming-distance/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。

> 给你两个整数 x 和 y，计算并返回它们之间的汉明距离。



```
按位异或后移位计数
```

- 异或
- 时间复杂度 $O(logC)$，C 是数字大小
- 空间复杂度 $O(1)$

```
异或后通过移位统计 1 的个数
```

```C++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int z = x^y;
        int ans = 0;
        while (z)
        {
            ans += (z&1);
            z >>= 1;
        }
        return ans;
    }
};
```



## [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一棵二叉树的根节点，返回该树的 直径 。

> 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

> 两节点之间路径的 长度 由它们之间边数表示。

```
![](https://file.fbichao.top/2024/03/6f14534edd42950e0c5db39bf1807479.png)
输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
```

- 先序遍历的递归形式
- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```
先序遍历递归式
```

```C++
class Solution {
private:
    int sum = 0;

public:
    // 统计节点的个数
    int dfs(TreeNode* root)
    {
        if (root == nullptr) return 0;

        // 左右子树节点个数
        int leftNum = dfs(root->left);
        int rightNum = dfs(root->right);

        // 结果需要 leftNum 和 rightNum
        sum = max(sum, 1 + leftNum + rightNum);

        // dfs 返回值表示从该 root 开始遍历，最有多少个节点
        // 所以只能加上 leftNum 和 rightNum 中最大的
        return 1 + max(leftNum, rightNum);
    }

    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        // 需要减 1
        return sum - 1;
    }
};
```
