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
## [144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/description/)

> 给你二叉树的根节点 root ，返回它节点值的 前序 遍历。

```
输入：root = [1,null,2,3]
输出：[1,2,3]
```

- 前序遍历
- 后续遍历的递归写法就是把根节点的加入放在后面
- 后续遍历的迭代写法，后续遍历是左右根，而前序是根左右，把后续反转，即根右左，迭代时候注意先入栈左后入栈右节点即可

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
        // base case
        if (root == nullptr) return;
  
        res.push_back(root->val);       // 根
        traversal(root->left);          // 左
        traversal(root->right);         // 右
    }

    vector<int> preorderTraversal(TreeNode* root) {
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
    vector<int> preorderTraversal(TreeNode* root) {
        if (root == nullptr) return {};
        vector<int> res;
        stack<TreeNode*> st;

        st.push(root);
        while (!st.empty())
        {
            TreeNode* node = st.top(); st.pop();
            res.push_back(node->val);                   // 根
            if (node->right) st.push(node->right);      // 入栈是右，出栈左先出
            if (node->left) st.push(node->left);        // 
        }

        return res;
    }
};
```

## [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/description/)

```
递归算法简单，修改根节点的位置即可
迭代算法，使用[先序遍历](#144-二叉树的前序遍历)的迭代方法，再反转
```

## [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

> 图示两个链表在节点 c1 开始相交：
> ![](https://file.fbichao.top/2024/03/e894bf9bd8faa691a3ce9afb894897f2.png)

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

## [202. 快乐数](https://leetcode.cn/problems/happy-number/description/)

> 编写一个算法来判断一个数 n 是不是快乐数。

> 「快乐数」 定义为：

> 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
> 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
> 如果这个过程 结果为 1，那么这个数就是快乐数。
> 如果 n 是 快乐数 就返回 true ；不是，则返回 false 。

```
输入：n = 19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

### 模拟

- 时间复杂度 $O(logn)$
- 空间复杂度 $O(logn)$

```
使用哈希表存储计算的结果，看是否出现循环或者 1
```

```C++
class Solution {
public:
    bool isHappy(int n) {
        // 存储计算结果
        unordered_set<int> uset;
        while (1)
        {
            uset.insert(n);
            int res = cal(n);
            if (res == 1) return true;
            else if (uset.count(res)) return false;
            // 更新
            n = res;
        }
    }

    int cal(int n)
    {
        // 计算 n 每个位置平方和
        int res = 0;
        while (n)
        {
            res += (n % 10) * (n % 10);
            n /= 10;
        }

        return res;
    }
};
```

### 双指针

- 时间复杂度 $O(logn)$
- 空间复杂度 $O(1)$

```
通过快慢双指针找到循环点
```

```C++
class Solution {
public:
    bool isHappy(int n) {
        int slow = n, fast = n;
        while (1)
        {
            slow = cal(slow);
            fast = cal(cal(fast));
            if (slow == fast) break;
        }
        return slow == 1;
    }

    int cal(int n)
    {
        // 计算 n 每个位置平方和
        int res = 0;
        while (n)
        {
            res += (n % 10) * (n % 10);
            n /= 10;
        }

        return res;
    }
};
```

## [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/description/)

> 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

![](https://file.fbichao.top/2024/03/8520d803fa76421848544c6edd984421.png)

```
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
```

- 删除链表节点
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
通过指向待删除节点的前一个节点，删除链表节点
```

```C++
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (head == nullptr) return head;
        // 虚拟头，便于删除头结点
        ListNode* dummpy = new ListNode(-1);
        dummpy->next = head;
      
        // 前一个节点
        ListNode* prev = dummpy;
        // 当前节点
        ListNode* cur = prev->next;
        while (cur)
        {
            if (cur->val == val)
            {   // 删除
                prev->next = cur->next;
                cur = prev->next;
            }
            else
            {   // 移动
                prev = prev->next;
                cur = cur->next;
            }
        }
        return dummpy->next;
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

## [222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/description/)

> 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

- 分离双指针递归
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
类似[对称二叉树](LeetCodeEasy01.md#101-对称二叉树)
使用两个指针，分别向两边遍历
如果某个子树是满二叉树，就返回该子树节点个数
如果不是满的，就+1，遍历左右子节点是否为满的
```

```C++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == nullptr) return 0;

        TreeNode* leftNode = root->left;
        int leftCount = 0;
        TreeNode* rightNode = root->right;
        int rightCount = 0;

        while (leftNode)
        {
            leftNode = leftNode->left;
            ++leftCount;
        }

        while (rightNode)
        {
            rightNode = rightNode->right;
            ++rightCount;
        }

        // 满二叉树 2^k-1
        if (leftCount == rightCount)
        {
            return (2 << leftCount) - 1;
        }

        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
```

## [225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/description/)

> 请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。

> 实现 MyStack 类：

> void push(int x) 将元素 x 压入栈顶。
> int pop() 移除并返回栈顶元素。
> int top() 返回栈顶元素。
> boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。

```
用两个队列实现，push 正常 push 进一个队列，弹出元素时，将队列中的元素移动到另一个队列中，只剩下一个元素

也可以用一个队列实现
```

```C++
class MyStack {
private:
    queue<int> que;

public:
    MyStack() {

    }
  
    void push(int x) {
        que.push(x);
    }
  
    int pop() {
        int n = que.size();
        --n;
        while (n--)
        {
            que.push(que.front());  que.pop();
        }
        int num = que.front();  que.pop();
        return num;
    }
  
    int top() {
        int num = this->pop();
        que.push(num);
        return num;
    }
  
    bool empty() {
        return que.empty();
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

## [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/description/)

> 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

> 实现 MyQueue 类：

> void push(int x) 将元素 x 推到队列的末尾
> int pop() 从队列的开头移除并返回元素
> int peek() 返回队列开头的元素
> boolean empty() 如果队列为空，返回 true ；否则，返回 false

```
用两个栈实现，一个栈用于在队列出列的时候使用，另一个就存储 push 的
```

```C++
class MyQueue {
private:
    stack<int> st1;
    stack<int> st2;

public:
    MyQueue() {

    }
  
    void push(int x) {
        st1.push(x);
    }
  
    int pop() {
        // st2 用于出
        if (st2.empty())
        {
            while (!st1.empty())
            {
                st2.push(st1.top());  
                st1.pop();
            }
        }
        int res = st2.top();  st2.pop();
        return res;
    }
  
    int peek() {
        if (st2.empty())
        {
            while (!st1.empty())
            {
                st2.push(st1.top());  
                st1.pop();
            }
        }
        return st2.top();
    }
  
    bool empty() {
        return (st1.empty() && st2.empty());
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

## [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/description/)

> 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

> 注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

```
输入: s = "anagram", t = "nagaram"
输出: true
```

- 哈希表
- 时间复杂度 $O(n)$
- 空间复杂度 $O(\Sigma)$

```C++
class Solution {
public:
    bool isAnagram(string s, string t) {
        // 存储
        unordered_map<char, int> umap;
        // 统计 s 字符串
        for (char c: s)
        {
            umap[c]++;
        }

        // 根据 c 字符串删除
        for (char c: t)
        {
            if (umap[c] <= 0)
            {
                return false;
            }
            umap[c]--;
        }

        // 如果还有字符
        for (auto kv: umap)
        {
            if (kv.second != 0)
            {
                return false;
            }
        }

        return true;
    }
};
```

## [257. 二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/description/)

> 给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。

> 叶子节点 是指没有子节点的节点。

- 先序遍历递归式
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n^2)$

```
先序遍历构造路径，直到叶子结点才 push
```

```C++
class Solution {
public:
    void construct(TreeNode* root, string path, vector<string>& res)
    {
        if (root != nullptr)
        {
            // 构造 path
            path += to_string(root->val);
            // 如果是叶子结点了，就 push
            if (root->left == nullptr && root->right == nullptr) res.push_back(path);
            else
            {
                // 如果不是叶子结点，需要 "->"
                path += "->";
                // 继续遍历左子树和右子树，遍历到叶子结点就会终止，不会产生回溯
                construct(root->left, path, res);
                construct(root->right, path, res);
            }
        }
    }

    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        // res 存储结果，path 存储单个路径
        construct(root, "", res);
        return res;
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

## [344. 反转字符串](https://leetcode.cn/problems/reverse-string/description/)

> 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

> 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

```
输入：s = ["h","e","l","l","o"]
输出：["o","l","l","e","h"]
```

- 双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
双指针反转
```

```C++
class Solution {
public:
    void reverseString(vector<char>& s) {
        int left = 0, right = s.size() - 1;
        while (left < right)
        {
            swap(s[left], s[right]);
            ++left;
            --right;
        }
    }
};
```

## [349. 两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/description/)

> 给定两个数组 nums1 和 nums2 ，返回 它们的 交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。

```
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]
```

### 哈希

- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n)$

```C++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> uset;
        vector<int> res;

        for (int c: nums1) uset.insert(c);

        for (int c: nums2)
        {
            if (uset.count(c))
            {
                res.push_back(c);
                uset.erase(c);
            }
        }

        return res;
    }
};
```

### 分离双指针+排序

- 时间复杂度 $O(nlogn + mlogm)$
- 空间复杂度 $O(logm + logn)$

```C++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        int length1 = nums1.size(), length2 = nums2.size();
        int index1 = 0, index2 = 0;
        vector<int> res;

        // 对排序后的两个数组，分离双指针
        while (index1 < length1 && index2 < length2)
        {
            int a = nums1[index1], b = nums2[index2];
            if (a == b)
            {   // 去重
                if (res.size() == 0 || a != res.back())
                {
                    res.push_back(a);
                }
                ++index1;
                ++index2;
            }
            else if (a < b)
            {
                ++index1;
            }
            else
            {
                ++index2;
            }
        }
        return res;
    }
};
```

## [383. 赎金信](https://leetcode.cn/problems/ransom-note/description/)

> 给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。

> 如果可以，返回 true ；否则返回 false 。

> magazine 中的每个字符只能在 ransomNote 中使用一次。

```
输入：ransomNote = "aa", magazine = "ab"
输出：false
```

- unoreder_map
- 时间复杂度 $O(n+m)$
- 空间复杂度 $O(\Sigma)$

```C++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> umap;
        // 统计 ransomNote 字符个数
        for (char c: ransomNote) umap[c]++;

        // 使用 magazine 组成
        for (char c: magazine)
        {
            if (umap[c] > 0)
            {
                umap[c]--;
            }
        }

        // 如果还存在非零的，就无法构成
        for (auto kv: umap)
        {
            if (kv.second != 0)
            {
                return false;
            }
        }

        return true;
    }
};
```

## [392. 判断子序列](https://leetcode.cn/problems/is-subsequence/description/)

> 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

> 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

> 进阶：

> 如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？



```
输入：s = "abc", t = "ahbgdc"
输出：true
```

```C++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int m = s.size(), n = t.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (s[i-1] == t[j-1])
                {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else
                {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        return dp[m][n] == s.size();
    }
};
```

## [404. 左叶子之和](https://leetcode.cn/problems/sum-of-left-leaves/description/)

> 给定二叉树的根节点 root ，返回所有左叶子之和。

- 前序遍历递归式
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
左叶子结点，要么是 root 的左节点，要么在 root 的右子树中的某个左节点
```

```C++
class Solution {
public:
    bool isLeafNode(TreeNode* node) 
    {   // 判断是否为叶子结点
        return !node->left && !node->right;
    }

    int dfs(TreeNode* root)
    {
        int ans = 0;
        // 左子树
        if (root->left)
        {
            // 如果是叶子结点，则加上 val，否则递归左子树
            if (isLeafNode(root->left)) ans += root->left->val;
            else ans += dfs(root->left);
        }
        // 右子树
        if (root->right)
        {   
            if (isLeafNode(root->right)) return ans;
            else ans += dfs(root->right);
        }

        return ans;
    }

    int sumOfLeftLeaves(TreeNode* root) {
        if (root == nullptr) return 0;
        return dfs(root);
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
输入：x = 1, y = 4
输出：2
解释：
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
上面的箭头指出了对应二进制位不同的位置。
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

## [459. 重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/description/)

> 给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。

```
输入: s = "abab"
输出: true
解释: 可由子串 "ab" 重复两次构成。
```

- KMP
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
计算 s 的 next 数组
- 如果是重复构成的，最后一个字母的 next 数组一定不是 0
- 如果是重复构成的，size - next[size-1] 就是重复字母的最小个数，一定可以被 size 整除
```

```C++
class Solution {
public:
    vector<int> geneNext(string& s)
    {
        vector<int> next(s.size(), 0);

        int left = 0;
        for (int right = 1; right < s.size(); ++right)
        {
            while (left > 0 && s[left] != s[right])
            {
                left = next[left - 1];
            }

            if (s[left] == s[right])
            {
                ++left;
            }
            next[right] = left;
        }
        return next;
    }

    bool repeatedSubstringPattern(string s) {
        vector<int> next = geneNext(s);
        int size = s.size();
        // 两个条件，缺一不可
        // 如果是重复构成的，最后一个字母的 next 数组一定不是 0
        // 如果是重复构成的，size - next[size-1] 就是重复字母的最小个数，一定可以被 size 整除
        if (next[size - 1] && size % (size - next[size-1]) == 0) return true;
        return false;
    }
};
```

## [496. 下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/description/)

> nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。

> 给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。

> 对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。

> 返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。

```
输入：nums1 = [4,1,2], nums2 = [1,3,4,2].
输出：[-1,3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 4 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
- 1 ，用加粗斜体标识，nums2 = [1,3,4,2]。下一个更大元素是 3 。
- 2 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
```

- 哈希表+单调栈
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
使用哈希表存储第一个数组的值和索引
对第二个数组使用单调栈，根据栈顶元素查找索引赋值
```

```C++
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> umap;
        // 记录 nums1 的 数字:索引
        for (int i = 0; i < nums1.size(); ++i) umap[nums1[i]] = i;

        stack<int> st;
        int n = nums1.size();
        vector<int> res(n, -1);

        for (int i = 0; i < nums2.size(); ++i)
        {   // 单调递增栈
            while (!st.empty() && nums2[i] > nums2[st.top()])
            {
                // 如果 map 包含栈顶元素
                if (umap.count(nums2[st.top()]))
                {   // 通过栈顶元素查找索引，赋值
                    res[umap[nums2[st.top()]]] = nums2[i];
                }
                st.pop();
            }
            st.push(i);
        }

        return res;
    }
};
```

## [501. 二叉搜索树中的众数](https://leetcode.cn/problems/find-mode-in-binary-search-tree/description/)

> 给你一个含重复值的二叉搜索树（BST）的根节点 root ，找出并返回 BST 中的所有 众数（即，出现频率最高的元素）。

> 如果树中有不止一个众数，可以按 任意顺序 返回。

- 中序遍历
- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```
中序遍历得到升序数组，遍历找到众数次数，再遍历加入到结果，也可以使用哈希表
为了节省空间
可以使用变量 count 记录当前数字的个数，maxCount 记录最大个数
如果 count 超过 maxCount 就情况结果重算
```

```C++
class Solution {
private:
    // 存储结果
    vector<int> res;
    // 记录当前遍历的数的 count
    int count = 0;
    // 记录当前最大 count
    int maxCount = 0;
    // 记录前一个节点
    TreeNode* pre = nullptr;

public:
    void dfs(TreeNode* cur)
    {
        if (cur == nullptr) return;

        dfs(cur->left);

        // 第一个数
        if (pre == nullptr) count = 1;
        // 连续相等
        else if (pre->val == cur->val) ++count;
        // 出现新的数字
        else count = 1;
        // 赋值 pre
        pre = cur;

        // 如果和最大计数相等，追加到 res
        if (count == maxCount) res.push_back(cur->val);

        // 如果大于最大计数，更新最大计数，清空 res，追加 res
        if (count > maxCount)
        {
            maxCount = count;
            res.clear();
            res.push_back(cur->val);
        }

        dfs(cur->right);
    }

    vector<int> findMode(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

## [509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/description/)

```C++
class Solution {
public:
    int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        vector<int> dp(n+1);
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; ++i)
        {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
};
```

## [530. 二叉搜索树的最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/description/)

> 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。

> 差值是一个正数，其数值等于两值之差的绝对值。

```
![](https://file.fbichao.top/2024/03/48d0cded37c01d3a11aa3277a680b10e.png)
输入：root = [4,2,6,1,3]
输出：1
```

```
中序遍历得到的是递增的，所以只需要比较相邻元素差值即可
```

### 中序遍历递归法

- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```C++
class Solution {
private:
    int res = INT_MAX;          // 差值
    TreeNode* pre = nullptr;    // 记录前一个元素

public:
    void dfs(TreeNode* root)
    {
        if (root == nullptr) return;

        // 左节点
        dfs(root->left);
        // 中间节点，如果有前一个节点，那么更新 res，没有的话，把当前节点赋值为 pre
        if (pre) res = min(res, abs(root->val - pre->val));
        pre = root;
        // 右节点
        dfs(root->right);
    }

    int getMinimumDifference(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

### 中序遍历迭代法

- 时间复杂度 $O(n)$
- 空间复杂度 $O(h)$

```C++
class Solution {
public:
    int getMinimumDifference(TreeNode* root) {
        stack<TreeNode*> st;
        // 记录前一个节点
        TreeNode* pre = nullptr;
        // 记录结果
        int res = INT_MAX;
        // 记录当前节点
        TreeNode* cur = root;

        while (cur || !st.empty())
        {
            while (cur)
            {
                st.push(cur);
                cur = cur->left;
            }

            // 当前节点
            cur = st.top(); st.pop();
            if (pre)    // 如果存在前一个节点
            {           // 更新 res
                res = min(res, abs(pre->val - cur->val));
            }
            // 将当前节点赋值为 pre
            pre = cur;
            cur = cur->right;
        }

        return res;
    }
};
```

## [541. 反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/description/)

> 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。

> 如果剩余字符少于 k 个，则将剩余字符全部反转。
> 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

```
输入：s = "abcdefg", k = 2
输出："bacdfeg"
```

- 循环反转
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
循环步长为 2k
```

```C++
class Solution {
public:
    // 反转字符串
    void reverse(string& s, int left, int right)
    {
        while (left < right)
        {
            swap(s[left++], s[right--]);
        }
    }

    string reverseStr(string s, int k) {
        // 每次 2k 个遍历
        for (int i = 0; i < s.size(); i+=2*k)
        {
            // 不足 k 个
            if ((i + k) > s.size()) reverse(s, i, s.size() - 1);
            // 超过 k 个
            else reverse(s, i, i + k - 1);
        }

        return s;
    }
};
```

## [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一棵二叉树的根节点，返回该树的 直径 。

> 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

> 两节点之间路径的 长度 由它们之间边数表示。

![](https://file.fbichao.top/2024/03/6f14534edd42950e0c5db39bf1807479.png)

```
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
