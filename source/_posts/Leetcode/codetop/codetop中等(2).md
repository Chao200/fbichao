---
title: codetop中等(2)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Codetop
excerpt: codetop中等(2)
math: true
date: 2024-04-03 21:45:00
---
## [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/description/)

- 回溯

```C++
class Solution {
private:
    vector<string> res;
    vector<string> path;

public:
    // 判断字符串是否合法
    bool isValid(string& s, int start, int end)
    {
        // 越界
        if (start > end) return false;

        // 首部为 0，且不止一个字符构成
        if (s[start] == '0' && end - start >= 1) return false;

        // 不大于 255
        int num = 0;
        while (start <= end)
        {
            num = num * 10 + s[start] - '0';
            ++start;
            if (num > 255) return false;
        }
        return true;
    }

    void backtrack(string& s, int index)
    {
        if (path.size() > 4) return;
        // 组装
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

        // 现在遍历的 index
        for (int i = index; i < s.size(); ++i)
        {
            if (isValid(s, index, i))
            {
                path.push_back(s.substr(index, i - index + 1));
                backtrack(s, i + 1);    // 下次遍历的 index
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

## [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/)

- 双串线性 dp

```C++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        // 长度
        int m = text1.size(), n = text2.size();

        // +1
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (text1[i-1] == text2[j-1])
                {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else
                {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        return dp[m][n];
    
    }
};
```

## [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/description/)

- 链表删除节点

```C++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* dummpy = new ListNode(-1);
        dummpy->next = head;

        // 待删除节点的前一个节点
        ListNode* prev = dummpy;
        ListNode* cur = dummpy->next;
    
        while (cur && cur->next)
        {
            ListNode* nxt = cur->next;

            if (cur->val == nxt->val)
            {
                while (cur && nxt && cur->val == nxt->val)
                {
                    cur = cur->next;
                    nxt = nxt->next;
                }

                cur = cur->next;
                prev->next = cur;
            }
            else
            {
                prev = cur;
                cur = cur->next;
            }
        }

        return dummpy->next;
    }
};
```

## [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/description/)

- 层序遍历

```C++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if (root == nullptr) return res;

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty())
        {
            int size = que.size();

            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = que.front(); que.pop();
                if (i == size - 1) res.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }

        return res;
    }
};
```

## [31. 下一个排列](https://leetcode.cn/problems/next-permutation/description/)

- 首先反向找到第一个升序对(i,j)，此时从 j 到 end 都是降序，在这些降序中找到第一个比 i 大的进行 swap，此时 j~end 是降序的，为了得到最小的下一个排列，反转 j~end [题解](https://leetcode.cn/problems/next-permutation/solutions/80560/xia-yi-ge-pai-lie-suan-fa-xiang-jie-si-lu-tui-dao-/)

```C++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return;

        // i 和 j 用于查找反向第一对升序
        // nums[i] < nums[j]，或者 i == 0 说明是单调递减的
        int i = n - 2, j = n - 1;
    
        // 查找第一对递增序列
        while (i >= 0 && nums[i] >= nums[j])
        {
            --i;
            --j;
        }

        // 此时从 j~k 是单调递减的，从里面找到第一个比 i 大的，再 swap
        int k = n - 1;
        if (i >= 0)
        {
            while (nums[i] >= nums[k])
            {
                --k;
            }
            swap(nums[i], nums[k]);
        }

        // 此时 i 位置的值已经比之前大了，已经找到了下一个排列
        // 但是了该排列是最小的，需要反转 j~end 使之成为升序，即为最小的下一个排列
        reverse(nums.begin() + j, nums.end());
    }
};
```

## [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/description/)

- 状态机

```C++
class Automaton
{
    string state = "start";
    unordered_map<string, vector<string>> table = 
    {
        {"start", {"start", "signed", "in_number", "end"}},
        {"signed", {"end", "end", "in_number", "end"}},
        {"in_number", {"end", "end", "in_number", "end"}},
        {"end", {"end", "end", "end", "end"}}
    };

    // 从状态机
    int get_col(char c)
    {
        if (isspace(c)) return 0;
        if (c == '+' || c == '-') return 1;
        if (isdigit(c)) return 2;
        return 3;
    }

public:
    int sign = 1;
    long long ans = 0;

    // 主状态机
    void get(char c)
    {
        state = table[state][get_col(c)];
        if (state == "in_number")
        {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
        }
        else if (state == "signed")
        {
            sign = c == '+' ? 1 : -1;
        }
    }
};

class Solution {
public:
    int myAtoi(string s) {
        Automaton autom;
        for (char c: s)
        {
            autom.get(c);
        }
        return autom.sign * autom.ans;
    }
};
```

## [148. 排序链表](https://leetcode.cn/problems/sort-list/description/)

- 链表归并排序

```C++
class Solution {
public:
    ListNode* merge(ListNode* left, ListNode* right)
    {
        ListNode* dummpy = new ListNode(-1);
        ListNode* cur = dummpy;
    
        while (left && right)
        {
            if (left->val > right->val)
            {
                cur->next = right;
                right = right->next;
            }
            else
            {
                cur->next = left;
                left = left->next;
            }
            cur = cur->next;
        }

        cur->next = left == nullptr ? right:left;
        return dummpy->next;
    }

    ListNode* mergeSort(ListNode* head)
    {
        if (head == nullptr || head->next == nullptr) return head;

        ListNode* slow = head, *fast = head->next;

        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* leftNode = head;
        ListNode* rightNode = slow->next;
        slow->next = nullptr;

        return merge(mergeSort(leftNode), mergeSort(rightNode));
    }

    ListNode* sortList(ListNode* head) {
        return mergeSort(head);
    }
};
```

## [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/description/)

- 先生成左括号，后右括号

```C++
class Solution {
private:
    vector<string> res;
    string path;

public:
    void backtrack(int n, int left, int right)
    {
        if (path.size() == 2 * n)
        {
            res.push_back(path);
            return;
        }

        if (left < right) return;

        if (left < n)
        {
            path.push_back('(');
            backtrack(n, left + 1, right);
            path.pop_back();
        }

        if (right < left)
        {
            path.push_back(')');
            backtrack(n, left, right + 1);
            path.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        backtrack(n, 0, 0);
        return res;
    }
};
```

## [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/description/)

- 链表模拟两数相加

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 新的虚拟头结点
        ListNode* sumHead = new ListNode(-1);
        // 用于存储节点
        ListNode* cur = sumHead;

        // 基数和进位
        int base = 0, carry = 0;
        while (carry || l1 || l2)
        {
            // 两个待计算的数，若为空节点，则返回 0
            int num1 = l1 == nullptr ? 0 : l1->val;
            int num2 = l2 == nullptr ? 0 : l2->val;
            // 新的 基数和进位
            base = (num1 + num2 + carry) % 10;
            carry = (num1 + num2 + carry) / 10;

            // 添加新节点
            cur->next = new ListNode(base);
            cur = cur->next;

            // 非空移动
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }

        return sumHead->next;
    }
};
```

## [165. 比较版本号](https://leetcode.cn/problems/compare-version-numbers/description/)

- 模拟，取出每个 . 之前的数字比较

```C++
class Solution {
public:
    int compareVersion(string version1, string version2) {
        int n = version1.size(), m = version2.size();
        int i = 0, j = 0;

        // 提取出每个 . 之前对应的数字
        // 如果一个比另一个短，就直接取 0 比较
        while (i < n || j < m)
        {
            int x = 0;  
            while (i < n && version1[i] != '.')
            {
                x = x * 10 + (version1[i] - '0');
                ++i;
            }
            ++i;    // 跳过 .

            int y = 0;
            while (j < m && version2[j] != '.')
            {
                y = y * 10 + (version2[j] - '0');
                ++j;
            }
            ++j;    // 跳过 .

            if (x != y)
            {
                return x > y ? 1 : -1;
            }
        }
        return 0;
    }
};
```

## [322. 零钱兑换](https://leetcode.cn/problems/coin-change/description/)

- 完全背包问题，先正序遍历背包，再正序遍历物品

```C++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // 记录组成 amount 大小的个数
        vector<int> dp(amount+1, amount+1);
    
        dp[0] = 0;

        // 遍历背包
        for (int i = 1; i <= amount; ++i)
        {
            // 遍历物品
            for (int j = 0; j < coins.size(); ++j)
            {
                // 如果可以构成
                if (coins[j] <= i)
                {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
};
```

## [78. 子集](https://leetcode.cn/problems/subsets/description/)

- 回溯

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& nums, int index)
    {   // 直接 push
        res.push_back(path);

        // 当前的索引
        for (int i = index; i < nums.size(); ++i)
        {
            path.push_back(nums[i]);
            backtrack(nums, i+1);   // 下次的
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        backtrack(nums, 0);
        return res;
    }
};
```

## [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)

- 前序和中序遍历

```C++
class Solution {
private:
    unordered_map<int, int> umap;

public:
    TreeNode* build(vector<int>& preorder, vector<int>& inorder, 
                int pre_l, int pre_r, int ind_l, int ind_r)
    {
        if (ind_l == ind_r) return nullptr;
        // 需要减去 ind_l
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

## [43. 字符串相乘](https://leetcode.cn/problems/multiply-strings/description/)

- 模拟

```C++
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") return "0";

        int n1 = num1.size(), n2 = num2.size();
        // 两数相乘，最多是 n1+n2 位
        vector<int> mult(n1+n2, 0);

        // 模拟乘法运算
        // 从低位开始遍历 num1
        for (int i = n1 - 1; i >= 0; --i)
        {
            int x = num1[i] - '0';
            // 从地位开始遍历 num2
            for (int j = n2 - 1; j >= 0; --j)
            {
                int y = num2[j] - '0';
                mult[i+j+1] += x * y;
            }
        }

        // 开始从地位开始进位
        for (int i = n1 + n2 - 1; i > 0; --i)
        {
            mult[i - 1] += mult[i] / 10;
            mult[i] %= 10;
        }

        // 构建结果
        string ans;
        int index = 0;
        // 对首尾单独处理
        if (mult[index]) ans.push_back(mult[index] + '0');

        // 剩下的位
        ++index;
        while (index < n1+n2)
        {
            ans.push_back(mult[index++] + '0');
        }

        return ans;
    }
};
```

## [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/description/)

- 双指针，先反转整个字符串，再去除多余空格，再反转每个空格隔开的单词

```C++
class Solution {
public:
    string reverseWords(string s) {
        // 先反转整个字符串 O(n)
        reverse(s.begin(), s.end());

        // 通过快慢双指针，删除多余空格
        int slow = 0, fast = 0;
        while (fast < s.size())
        {   // 找到首个非空字符
            if (s[fast] != ' ')
            {
                // 在下次遍历的单词前加上空格
                if (slow != 0) s[slow++] = ' ';

                // 直到遇到空格停止
                while (fast < s.size() && s[fast] != ' ')
                {
                    s[slow++] = s[fast++];
                }
            }
            ++fast;
        }
      
        // resize
        s.resize(slow);
      
        // 双指针反转每个单词
        int left = 0, right = 0;
        while (right < s.size())
        {
            if (s[right] == ' ')
            {
                reverse(s.begin()+left, s.begin()+right);
                left = right + 1;
            }
            ++right;
        }

        // 反转最后一个单词
        reverse(s.begin() + left, s.end());
        return s;
    }
};
```

## [129. 求根节点到叶节点数字之和](https://leetcode.cn/problems/sum-root-to-leaf-numbers/description/)

- 先序遍历的递归式

```C++
class Solution {
public:
    // 先序遍历的递归形式
    int dfs(TreeNode* root, int preSum)
    {
        if (root == nullptr) return 0;
        int sum = preSum * 10 + root->val;
        if (root->left == nullptr && root->right == nullptr) return sum;
        return dfs(root->left, sum) + dfs(root->right, sum);
    }

    int sumNumbers(TreeNode* root) {
        return dfs(root, 0);
    }
};
```

## [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/)

- 中序遍历是单调递增的

```C++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* cur = root;
        long prev = LONG_MIN;
      
        while (!st.empty() || cur)
        {
            while (cur)
            {
                st.push(cur);
                cur = cur->left;
            }

            cur = st.top(); st.pop();

            if (prev >= cur->val) return false;

            prev = cur->val;
            cur = cur->right;
        }

        return true;
    }
};
```

## [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/description/)

- 动态规划

```C++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        vector<vector<int>> dp(rows, vector<int>(cols, 0));
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; ++i) dp[i][0] = dp[i-1][0] + grid[i][0];
        for (int i = 1; i < cols; ++i) dp[0][i] = dp[0][i-1] + grid[0][i];

        for (int i = 1; i < rows; ++i)
        {
            for (int j = 1; j < cols; ++j)
            {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + + grid[i][j];
            }
        }

        return dp[rows-1][cols-1];
    }
};
```

## [48. 旋转图像](https://leetcode.cn/problems/rotate-image/description/)

- 两次对称变换

```C++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        // 对角线对换
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        // 左右对称变换
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < (cols / 2); ++j)
            {
                swap(matrix[i][j], matrix[i][cols-j-1]);
            }
        }
    }
};
```

## [470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7/description/)

- [题解](https://leetcode.cn/problems/implement-rand10-using-rand7/solutions/427572/cong-pao-ying-bi-kai-shi-xun-xu-jian-jin-ba-zhe-da/)

```C++
class Solution {
public:
    int rand10() {
        while (1)
        {
            // x 范围是 [0, 6] * 7 = 0, 7, 14, ……, 42
            // 两两之间间隔 6，为了补齐缺失的数，加上 rand7() - 1 即可
            int x = (rand7() - 1) * 7 + (rand7() - 1);
            if (x >= 1 && x<= 40)
            {
                return x % 10 + 1;
            }

            // 0 和 41,42,43,44,45,46,47,48 没利用上，可以对 40 取模
            // 得到 [0, 8] * 7 = 0, 7, ……, 56 间隔也是 6，补齐
            x = (x % 40) * 7 + rand7() - 1;
            if (x <= 60) return x % 10 + 1;

            // 最后
            x = (x - 61) * 7 + rand7() - 1;
            if (x < 20) return x % 10 + 1;
        }
    }
};
```
