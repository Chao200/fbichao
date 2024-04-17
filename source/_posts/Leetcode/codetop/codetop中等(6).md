---
title: codetop中等(6)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Codetop
excerpt: codetop中等(6)
math: true
date: 2024-04-13 21:45:00
---

## [189. 轮转数组](https://leetcode.cn/problems/rotate-array/description/)

- 反转三次

```C++
class Solution {
public:
    void reverse(vector<int>& nums, int left, int right)
    {
        while (left < right)
        {
            swap(nums[left++], nums[right--]);
        }
    }

    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        reverse(nums, 0, nums.size() - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.size() - 1);
    }
};
```



## [442. 数组中重复的数据](https://leetcode.cn/problems/find-all-duplicates-in-an-array/description/)

- 原地哈希

```C++
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        int len = nums.size();

        // 原地哈希交换
        for (int i = 0; i < len; ++i)
        {
            // 直到相等，才继续下一个元素
            while (nums[i] != nums[nums[i]-1])
            {
                swap(nums[i], nums[nums[i]-1]);
            }
        }

        vector<int> res;
        for (int i = 0; i < len; ++i)
        {
            if (nums[i] != (i+1))
            {
                res.push_back(nums[i]);
            }
        }

        return res;
    }
};
```


## [120. 三角形最小路径和](https://leetcode.cn/problems/triangle/description/)

- 动态规划

```C++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        vector<vector<int>> dp(n, vector<int>(n));

        dp[0][0] = triangle[0][0];
        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = dp[i-1][0] + triangle[i][0];
            for (int j = 1; j < i; ++j)
            {
                dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j];
            }
            dp[i][i] = dp[i-1][i-1] + triangle[i][i];
        }
        
        int ans = dp[n-1][0];
        for (int i = 1; i < n; ++i)
        {
            ans = min(ans, dp[n-1][i]);
        }
        return ans;
    }
};
```


## [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/description/)

- 动态规划，固定根节点，讨论左右子树个数

```C++
class Solution {
public:
    int numTrees(int n) {
        // 三个节点
        // 以 3 开头，左边有两个比他小 dp[2]，右边没有 d[0]
        // 以 1 开头，左边没有 dp[0]，右边 dp[2]
        // 以 2 开头，左边 dp[1]，右边 dp[1]
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


## [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)

- 先找到根节点，再根据根节点划分左右子树

```C++
class Solution {
private:
    unordered_map<int, int> index;

public:
    TreeNode* dfs(vector<int> &inorder, vector<int> &postorder, 
                        int in_l, int in_r, int post_l, int post_r)
    {
        if (in_l == in_r) return nullptr;

        // 左子树的大小，需要减去 in_l
        int left_size = index[postorder[post_r - 1]] - in_l;
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


## [36进制加法](https://mp.weixin.qq.com/s/XcKQwnwCh5nZsz-DLHJwzQ)

36进制由0-9，a-z，共36个字符表示。

要求按照加法规则计算出任意两个36进制正整数的和，如：1b + 2x = 48  （解释：47+105=152）

要求：不允许使用先将36进制数字整体转为10进制，相加后再转回为36进制的做法

- 模拟

```C++
#include <iostream>
#include <algorithm>
using namespace std;

// int 转 char
char getChar(int n)
{
    if (n <= 9)
        return n + '0';
    else
        return n - 10 + 'a';
}

// char 转 int
int getInt(char ch)
{
    if ('0' <= ch && ch <= '9')
        return ch - '0';
    else
        return ch - 'a' + 10;
}

// 36 进制加法
string add36Strings(string num1, string num2)
{
    int carry = 0;
    int i = num1.size() - 1, j = num2.size() - 1;
    int x, y;
    string res;
    while (i >= 0 || j >= 0 || carry)
    {
        x = i >= 0 ? getInt(num1[i]) : 0;
        y = j >= 0 ? getInt(num2[j]) : 0;
        int temp = x + y + carry;
        res += getChar(temp % 36);
        carry = temp / 36;
        i--, j--;
    }
    reverse(res.begin(), res.end());
    return res;
}

int main()
{
    string a = "1b", b = "2x", c;
    c = add36Strings(a, b);
    cout << c << endl;
}
```



## [611. 有效三角形的个数](https://leetcode.cn/problems/valid-triangle-number/description/)

- 排序 + 双指针

```C++
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());

        int ans = 0;
        for (int i = 0; i < n; ++i)
        {
            int fast = i + 2;
            // slow 固定，只有 fast 动，所以该层循环是 O(n)
            for (int slow = i + 1; slow < n; ++slow)
            {
                while (fast < n && nums[i] + nums[slow] > nums[fast])
                {
                    ++fast;
                }
                ans += max(fast - slow - 1, 0);
            }
        }
        return ans;
    }
};
```

## [400. 第 N 位数字](https://leetcode.cn/problems/nth-digit/description/)

- 模拟

```C++
class Solution {
public:
    // 一位数的个数 1 * 9 个
    // 两位数的个数 2 * 9 * 10 个
    // 三位数的个数 3 * 9 * 100 个
    // …………
    int findNthDigit(int n) {
        long digit = 1;
        long base = 1;
        long count = 9;
        while (n > count)
        {
            n -= count;
            digit += 1;
            base *= 10;
            count = digit * base * 9;
        }

        // 10 11 12 13
        // 每个数字占用 digit = 2 位
        // 所以除以 2 得到的就是第几个数字了
        // (n-1)/digit 可以得到是哪一个数字
        long num = base + (n - 1) / digit;
        // 从左到右
        return to_string(num)[(n-1)%digit] - '0';
    }
};
```

## [面试题 02.05. 链表求和](https://leetcode.cn/problems/sum-lists-lcci/description/)

- 低位到高位存储的

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* newNode = new ListNode(-1);
        ListNode* cur = newNode;

        int base = 0, carry = 0;
        while (l1 || l2 || carry)
        {
            int n1 = l1 ? l1->val : 0;
            int n2 = l2 ? l2->val : 0;

            int sum = n1 + n2 + carry;
            base = sum % 10;
            carry = sum / 10;
            cur->next = new ListNode(base);
            cur = cur->next;

            l1 = l1 ? l1->next : nullptr;
            l2 = l2 ? l2->next : nullptr;
        }
        return newNode->next;
    }
};
```

## [71. 简化路径](https://leetcode.cn/problems/simplify-path/description/)

- 先 split，再使用栈模拟

```C++
class Solution {
public:
    vector<string> split(const string& s, char delim)
    {
        vector<string> ans;
        string cur;
        for (char c: s)
        {
            if (c == delim)
            {
                ans.push_back(cur);
                cur = "";
            }
            else
            {
                cur += c;
            }
        }
        ans.push_back(cur);
        return ans;
    }

    string simplifyPath(string path) {
        vector<string> names = split(path, '/');
        stack<string> st;
        for (auto name: names)
        {
            if (name == "..")
            {
                if (!st.empty())
                {
                    st.pop();
                }
            }
            else if (!name.empty() && name != ".")
            {
                st.push(name);
            }
        }

        string ans;
        if (st.empty())
        {
            ans = "/";
        }
        else
        {
            while (!st.empty())
            {
                auto name = st.top(); st.pop();
                ans = "/" + name + ans;
            }
        }

        return ans;
    }
};
```


## [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/description/)

- 滑动窗口，right 一直动，left 根据窗口内 0 的个数来决定是否移动

```C++
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int n = nums.size();
        int res = 0;
        int zero_num = 0;
        int left = 0;
        // right 一直动
        // 如果窗口内的 0 个数超过 k
        // 就移动 left，直到个数为 k 个才停止 left
        for (int right = 0; right < n; ++right)
        {
            if (nums[right] == 0) ++zero_num;
            // 如果 0 的个数多了，left 移动，直到 0 的个数合理
            while (zero_num > k)
            {
                if (nums[left++] == 0) --zero_num;
            }
            // 更新 res
            res = max(res, right - left + 1);
        }
        return res;
    }
};
```



## [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/description/)

- 动态规划

```C++
class Solution {
public:
    int numSquares(int n) {
        // 返回的是最少数量，所以需要取 MAX
        vector<int> dp(n+1, INT_MAX);
        dp[0] = 0;

        // 从 1~n-1 取出多个元素构成，可重复，组合问题
        // 先遍历物品
        for (int i = 1; i <= n; ++i)
        {
            int temp = i * i;
            if (temp > n) break;

            // 再遍历背包
            // 一个数是 temp，一个数是 j-temp
            for (int j = temp; j <= n; ++j)
            {
                dp[j] = min(dp[j - temp] + 1, dp[j]);
            }
        }
        

        return dp[n] == INT_MAX ? 0 : dp[n];
    }
};
```

## [678. 有效的括号字符串](https://leetcode.cn/problems/valid-parenthesis-string/description/)

- 辅助栈

```C++
class Solution {
public:
    bool checkValidString(string s) {
        // 辅助栈
        // 必须存索引，无法判断括号和 * 号谁先入栈的
        stack<int> st;
        stack<int> sup;
        int n = s.size();

        for (int i = 0; i < n; ++i)
        {
            char c = s[i];
            if (c == '(')
            {
                st.push(i);
            }
            else if (c == '*')
            {
                sup.push(i);
            }
            else
            {
                if (!st.empty())
                {
                    st.pop();
                }
                else if (!sup.empty())
                {
                    sup.pop();
                }
                else
                {
                    return false;
                }
            }
        }

        while (!st.empty() && !sup.empty())
        {
            int stIndex = st.top(); st.pop();
            int supIndex = sup.top(); sup.pop();
            if (stIndex > supIndex)
            {
                return false;
            }
        }

        return st.empty();
    }
};
```


- 贪心策略

```C++
class Solution {
public:
    // 贪心策略
    // 如果出现左括号，则未匹配的左括号数+1
    // 如果出现右括号，则未匹配的左括号数-1
    // 如果出现 *，未匹配的左括号数可能 +1 -1 不变
    // 维护一个左括号的最小和最大数量
    bool checkValidString(string s) {
        int minCount = 0, maxCount = 0;
        int n = s.size();
        for (int i = 0; i < n; ++i)
        {
            char c = s[i];
            if (c == '(')
            {
                minCount++;
                maxCount++;
            }
            else if (c == ')')
            {
                // 最小值不为负数，不然就错了
                minCount = max(minCount - 1, 0);
                maxCount--;
                // 右括号太多了
                if (maxCount < 0) return false;
            }
            else
            {
                minCount = max(minCount - 1, 0);
                maxCount++;
            }
        }
        return minCount == 0;
    }
};
```



## [673. 最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/description/)

- 动态规划，求方案数，单串分割

```C++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        int maxLen = 0, ans = 0;
        vector<int> dp(n, 1);
        vector<int> cnt(n, 1);

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                if (nums[j] < nums[i])
                {
                    // 构成更长的子序列
                    if (dp[j] + 1 > dp[i])
                    {
                        dp[i] = dp[j] + 1;  // 更新
                        cnt[i] = cnt[j];    // 重置为新的
                    }
                    // 包含和不包含 j 的子序列个数一样
                    else if (dp[j] + 1 == dp[i])
                    {
                        cnt[i] += cnt[j];   // 累和
                    }
                }
            }

            if (dp[i] > maxLen) // 更新新的长度和计数
            {
                maxLen = dp[i];
                ans = cnt[i];
            }
            else if (dp[i] == maxLen)   // 如果长度一致，则累加
            {
                ans += cnt[i];
            }
        }
        return ans;
    }
};
```

## [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/description/)

- 矩形 dp

```C++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));

        // 初始化第一行第一列
        for (int i = 0; i < m; ++i) 
        {
            if (obstacleGrid[i][0]) break;
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; ++i)
        {
            if (obstacleGrid[0][i]) break;
            dp[0][i] = 1;
        }

        // 遍历剩余的
        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                if (obstacleGrid[i][j]) 
                {
                    dp[i][j] = 0;
                    continue;
                }
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```


## [134. 加油站](https://leetcode.cn/problems/gas-station/description/)

- 贪心，如果从某地出发，在 B 点停止，则下次从 B 的下一个位置继续遍历

```C++
class Solution {
public:
    // 从 0 开始出发，如果可以走一圈，那结束
    // 如果不可以，停在了 z，则下一次从 z+1 开始遍历即可
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();

        // 表示走到哪个加油站
        int i = 0;
        while (i < n)
        {
            // 求走过的加油站总汽油和总花费
            int sumGas = 0, sumCost = 0;
            int cnt = 0;    // 计数走过的加油站
            while (cnt < n)
            {
                int j = (i + cnt) % n;  // i+cnt 表示第几个加油站，可能越界需要求余
                sumGas += gas[j];
                sumCost += cost[j];
                if (sumGas < sumCost) break;    // 停止
                ++cnt;
            }

            if (cnt == n)   // 如果走过的加油站个数是 n，则走得通
            {
                return i;
            }
            else    // 否则继续换下一个起始为止
            {
                i = i + cnt + 1;    // 这里不会越界，因为肯定是从 0~n-1 出发
            }
        }
        return -1;
    }
};
```



## [97. 交错字符串](https://leetcode.cn/problems/interleaving-string/description/)

- 三串动态规划

```C++
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() + s2.size() != s3.size()) return false;

        // dp[i][j] s1 的前 i 个和 s2 的前 j 个，是否可以构成 s3 的 i+j 个
        vector<vector<bool>> dp(s1.size() + 1, vector<bool>(s2.size() + 1, false));

        dp[0][0] = true;
        // 先看两个单串
        for (int i = 1; i <= s1.size(); ++i)
        {
            dp[i][0] = dp[i-1][0] && s1[i-1]==s3[i-1];
        }
        for (int i = 1; i <= s2.size(); ++i)
        {
            dp[0][i] = dp[0][i-1] && s2[i-1]==s3[i-1];
        }

        // 再看双串
        for (int i = 1; i <= s1.size(); ++i)
        {
            for (int j = 1; j <= s2.size(); ++j)
            {   // 可能是 s1 贡献，也可能是 s2 贡献的
                dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[i+j-1])
                || (dp[i][j-1] && s2[j-1] == s3[i+j-1]);
            }
        }
        return dp[s1.size()][s2.size()];
    }
};
```


## [264. 丑数 II](https://leetcode.cn/problems/ugly-number-ii/description/)

- 动态规划

```C++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n+1);
        dp[1] = 1;

        int two = 1;
        int three = 1;
        int five = 1;

        for (int i = 2; i <= n; ++i)
        {
            // dp[i] 是由前面较小的丑数乘以 2 3 5 得到的，取 min
            dp[i] = min(dp[two] * 2, min(dp[three] * 3, dp[five] * 5));
            // 比对看是哪个数得到的，索引就往前动
            if (dp[i] == dp[two] * 2) ++two;
            if (dp[i] == dp[three] * 3) ++three;
            if (dp[i] == dp[five] * 5) ++five;
        }

        return dp[n];
    }
};
```

## [LCR 153. 二叉树中和为目标值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/description/)

- 回溯

```C++
class Solution {
private:
    vector<int> path;
    vector<vector<int>> res;

public:
    void dfs(TreeNode* root, int target, int sum)
    {
        if (root == nullptr) return;

        path.push_back(root->val);
        sum += root->val;
        if (root->left == nullptr && root->right == nullptr)
        {
            if (sum == target)
            {
                res.push_back(path);
            }
        }
        dfs(root->left, target, sum);
        dfs(root->right, target, sum);
        sum -= root->val;
        path.pop_back();
    }

    vector<vector<int>> pathTarget(TreeNode* root, int target) {
        dfs(root, target, 0);
        return res;
    }
};
```
