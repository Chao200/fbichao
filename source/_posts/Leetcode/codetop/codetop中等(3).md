---
title: codetop中等(3)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Codetop
excerpt: codetop中等(3)
math: true
date: 2024-04-05 21:45:00
---

## [39. 组合总和](https://leetcode.cn/problems/combination-sum/description/)

- 回溯

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& candidates, int target, int sum, int index)
    {
        if (sum > target) return;
        if (sum == target)
        {
            res.push_back(path);
            return;
        }

        // 从 i 开始遍历满足条件的
        for (int i = index; i < candidates.size(); ++i)
        {
            sum += candidates[i];
            path.push_back(candidates[i]);
            backtrack(candidates, target, sum, i);  // 遍历 i 满足条件的
            path.pop_back();
            sum -= candidates[i];
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtrack(candidates, target, 0, 0);
        return res;
    }
};
```

## [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/description/)

- 回溯

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtracking(TreeNode* root, int targetSum)
    {
        if (root == nullptr) return;

        path.push_back(root->val);
        targetSum -= root->val;
        if (root->left == nullptr && root->right == nullptr && targetSum == 0)
        {
            res.push_back(path);
        }

        backtracking(root->left, targetSum);
        backtracking(root->right, targetSum);
        path.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        backtracking(root, targetSum);
        return res;
    }
};
```


## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

- 二分查找，左闭右闭查找边界

```C++
class Solution {
public:
    int binarySearchLeft(vector<int>& nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target)
            {
                right = mid - 1;
            }
            else if (nums[mid] < target)
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }

        if (left < 0 || left >= nums.size()) return -1;
        return nums[left] == target ? left : -1;
    }

    int binarySearchRight(vector<int>& nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target)
            {
                right = mid - 1;
            }
            else if (nums[mid] < target)
            {
                left = mid + 1;
            }
            else
            {
                left = mid + 1;
            }
        }

        if (right < 0 || right >= nums.size()) return -1;

        return nums[right] == target ? right : -1;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.size() == 0) return {-1, -1};
        int left = binarySearchLeft(nums, target);
        int right = binarySearchRight(nums, target);
        return {left, right};
    }
};
```

## [394. 字符串解码](https://leetcode.cn/problems/decode-string/description/)

- 栈模拟

```C++
class Solution {
public:
    // s 字符串重复 n 次
    string repeat(string s, int n)
    {
        string ans = "";
        while (n--) ans += s;
        return ans;
    }

    string decodeString(string s) {
        // 存储数字之前的字符串
        stack<string> str_st;
        // 存储当前数字
        stack<int> num_st;
        string res = "";
        int cur_num = 0;

        for (char c: s)
        {
            if (c >= '0' && c <= '9')
            {
                cur_num = cur_num * 10 + c - '0';
            }
            else if (c == '[')
            {
                // 数字之前的字符串
                str_st.push(res);
                // 当前数字
                num_st.push(cur_num);
                res = "";
                cur_num = 0;
            }
            else if (c == ']')
            {
                string str = str_st.top(); str_st.pop();
                int num = num_st.top(); num_st.pop();
                res = str + repeat(res, num);
            }
            else 
            {
                res += c;
            }
        }

        return res;
    }
};
```


## [221. 最大正方形](https://leetcode.cn/problems/maximal-square/description/)

- 动态规划

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
                    // base case
                    if (i==0||j==0) dp[i][j] = 1;
                    else dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1;
                }
                res = max(res, dp[i][j]);
            }
        }
        return res * res;
    }
};
```

## [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/)

- 右上角往左下角搜索

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


## [162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/description/)

- 二分搜索

```C++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left < right)
        {
            int mid = left + (right - left) / 2;
            // 左侧单调递增，峰值在右侧
            if (nums[mid] < nums[mid + 1])
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }
        return left;
    }
};
```

## [718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/description/)

- 双串动态规划

```C++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        int res = 0;

        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (nums1[i-1] == nums2[j-1])
                {
                    dp[i][j] = dp[i-1][j-1] + 1;
                    res = max(res, dp[i][j]);
                }
            }
        }

        return res;
    }
};
```

## [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)

- 哈希表

```C++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> uset;
        for (int num: nums) uset.insert(num);

        int res = 0;

        for (auto num: uset)
        {
            // 找到起始元素
            if (!uset.contains(num - 1))
            {
                // 以该起始元素开始计数和长度
                int currNum = num;
                int currLen = 1;

                while (uset.contains(currNum + 1))
                {
                    currNum++;
                    currLen++;
                }
                
                // 更新
                res = max(res, currLen);
            }
        }

        return res;
    }
};
```

## [662. 二叉树最大宽度](https://leetcode.cn/problems/maximum-width-of-binary-tree/description/)

- 层次遍历，存储节点和对应的序号

```C++
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        queue<pair<TreeNode*, long long>> que;

        que.push({root, 1});

        long long res = 1;

        while (!que.empty())
        {
            int n = que.size();

            long long first_index = 0, last_index = 0;

            for (int i = 0; i < n; ++i)
            {
                auto node = que.front(); que.pop();

                // 第一个节点的索引是开始
                if (i == 0) first_index = node.second;
                // 最后一个节点的索引是结束
                if (i == n - 1) last_index = node.second;
                                                                    // 左节点存在，索引是 父节点的索引 * 2
                if (node.first->left) que.push({node.first->left, (node.second - first_index) * 2});
                                                                    // 左节点存在，索引是 父节点的索引 * 2 + 1
                if (node.first->right) que.push({node.first->right, (node.second - first_index) * 2 + 1});
            }
            res = max(res, last_index - first_index + 1);
        }

        return res;
    }
};
```


## [62. 不同路径](https://leetcode.cn/problems/unique-paths/description/)

- 二维动规

```C++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));

        dp[0][0] = 1;
        for (int i = 0; i < m; ++i) dp[i][0] = 1;
        for (int i = 0; i < n; ++i) dp[0][i] = 1;

        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }

        return dp[m-1][n-1];
    }
};
```

## [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/)

- 动态规划，考虑多个负数相乘为正数

```C++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        // 需要考虑多个负数相乘转为正数
        // 同时记录 max 和 min
        vector<int> dp_min(n, 0);
        vector<int> dp_max(n, 0);

        dp_min[0] = nums[0];
        dp_max[0] = nums[0];
        int res = nums[0];

        for (int i = 1; i < n; ++i)
        {
            dp_min[i] = min({dp_min[i-1] * nums[i], nums[i], dp_max[i-1]*nums[i]});
            dp_max[i] = max({dp_min[i-1] * nums[i], nums[i], dp_max[i-1]*nums[i]});

            res = max(res, dp_max[i]);
        }
        return res;
    }
};
```

## [695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/description/)

- 深度优先搜索

```C++
class Solution {
private:
    int sum = 0;

public:
    void dfs(vector<vector<int>>& grid, int row, int col, vector<vector<bool>>& visited)
    {
        if (row < 0 || row >= grid.size()) return;
        if (col < 0 || col >= grid[0].size()) return;
        if (visited[row][col] || grid[row][col] == 0) return;

        visited[row][col] = true;
        ++sum;
        dfs(grid, row - 1, col, visited);
        dfs(grid, row + 1, col, visited);
        dfs(grid, row, col - 1, visited);
        dfs(grid, row, col + 1, visited);
    }

    int maxAreaOfIsland(vector<vector<int>>& grid) {
        vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
        int ans = 0;
        for (int row = 0; row < grid.size(); ++row)
        {
            for (int col = 0; col < grid[0].size(); ++col)
            {
                if (!visited[row][col] && grid[row][col] == 1)
                {
                    sum = 0;
                    dfs(grid, row, col, visited);
                    ans = max(ans, sum);
                }
            }
        }
        return ans;
    }
};
```

## [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/description/)

- 模拟，非栈

```C++
class Solution {
public:
    bool isdigit(char c)
    {
        return c >= '0' && c <= '9';
    }

    int calculate(string s) {
        vector<int> st;
        int n = s.size();
        int index = 0;
        char op = '+';
        while (index < n)
        {
            if (s[index] == ' ')
            {
                ++index;
                continue;
            }

            if (isdigit(s[index]))
            {
                // 数字
                int num = s[index] - '0';
                while (index + 1 < n && isdigit(s[index+1]))
                {
                    ++index;
                    num = 10 * num + (s[index] - '0');
                }

                // 数字前面的符号
                switch (op)
                {
                    // 加号和负号，直接入栈
                    case '+':
                        st.push_back(num);
                        break;
                    case '-':
                        st.push_back(-num);
                        break;
                    // 乘除法，计算后入栈
                    case '*':
                        st.back() *= num;
                        break;
                    case '/':
                        st.back() /= num;
                        break;
                }
            }
            // 更改符号
            else op = s[index];

            ++index;
        }
        
        // 求和
        int res = 0;
        for (auto n: st) res += n;
        return res;
    }
};
```


## [179. 最大数](https://leetcode.cn/problems/largest-number/description/)

- 贪心，如果字符串 ab 比 ba 大，则 a 在 b 之前

```C++
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        vector<string> str;
        for (auto num: nums) str.push_back(to_string(num));

        auto cmp = [](string left, string right)
        {
            return left + right > right + left;
        };

        sort(str.begin(), str.end(), cmp);

        string ans;
        for (auto s: str) ans += s;
        if (ans[0] == '0') return "0";
        return ans;
    }
};
```

## [139. 单词拆分](https://leetcode.cn/problems/word-break/description/)

- 完全背包问题，动规，单串线性 dp 分阶段

```C++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> uset(wordDict.begin(), wordDict.end());
        vector<bool> dp(s.size()+1, false);
        dp[0] = true;

        for (int i = 1; i <= s.size(); ++i) // 背包
        {
            for (int j = 0; j < i; ++j) // 物品
            {
                // 拆分从 0--j--i
                string word = s.substr(j, i-j); // 物品
                if (dp[j] && uset.find(word) != uset.end())
                {
                    dp[i] = true;
                }
            }
        }

        return dp[s.size()];
    }
};
```

## [912. 排序数组](https://leetcode.cn/problems/sort-an-array/description/)

- 见[归并、快速、堆排序](codetop中等(1).md#912-排序数组)


## [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/)

- 链表，四个指针，分别指向两个交换的节点以及它们的前后节点

```C++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;

        ListNode* dummpy = new ListNode(-1);
        dummpy->next = head;

        // 交换的前一个节点
        ListNode* prev = dummpy;
        // 待交换的第一个节点
        ListNode* cur = prev->next;
        while (cur && cur->next)
        {
            // 待交换的第二个节点
            ListNode* nxt = cur->next;
            // 后续需要交换的节点
            ListNode* nnxt = nxt->next;
            nxt->next = cur;
            cur->next = nnxt;
            prev->next = nxt;

            prev = cur;
            cur = nnxt;
        }

        return dummpy->next;
    }
};
```


## [198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/)

- 动态规划，偷或者不偷当前的

```C++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();

        if (n == 0) return 0;
        if (n == 1) return nums[0];

        vector<int> dp(n, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);

        for (int i = 2; i < n; ++i)
        {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }

        return dp[n-1];
    }
};
```

## [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/description/)

- 滑动窗口

```C++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        // 滑动窗口
        int slow = 0, fast = 0;

        int sum = 0;
        int res = nums.size() + 1;

        while (fast < nums.size())
        {
            // 求窗口内的和
            sum += nums[fast];
            while (sum >= target)
            {   // 如果和大于满足条件
                res = min(res, fast - slow + 1);
                sum -= nums[slow++];
            }
            ++fast;
        }

        return res == nums.size() + 1 ? 0 : res;
    }
};
```